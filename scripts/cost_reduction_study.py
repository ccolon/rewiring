"""Cost-reduction (theta_T) study, parameterised by visibility tier τ.

Mirrors visibility_study.py's loop structure but tracks cost outcomes (not
diversity). Two CSVs are written per cell:

  cost_reduction_trial.csv  -- one row per (tech_seed, init_seed) trial
  cost_reduction_firm.csv   -- one row per (tech_seed, init_seed, firm_idx)

For homogeneous τ cells, tier_i is identical across firms in a trial. For
heterogeneous τ cells (tier_std > 0), tier_i is drawn lognormal-style and
varies per firm — the firm-level CSV exposes the within-trial variation.

Cost-reduction window: aggregate sum_p averaged over the last
K = min(10, total_rounds // 2) rounds. This handles cycling regimes by
averaging over the cycle.

Usage (matches visibility_study.py CLI):
    python scripts/cost_reduction_study.py \\
        --n 50 --tech_per_job 5 --nb_rounds 200 \\
        --cc 4 --max_swaps 1 \\
        --aisi_spread 0.0 --sigma_w 0.0 \\
        --a_config homogeneous:0.5 --b_config homogeneous:0.9 --z_config homogeneous:1.0 \\
        --tau_values 0,1,2,3,4,5,6 --tau_mode homo \\
        --base_seed 1100 \\
        --output results/cost/cost_n50_oldpaper_seed1100.csv
"""
import argparse
import csv
import os
import sys
from itertools import combinations

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.networks import (
    build_W_from_suppliers,
    generate_base_network,
    generate_random_initial_network,
)
from rewiring.equilibrium import compute_adjusted_z, compute_equilibrium_full
from rewiring.simulation import run_unified_simulation


CYCLE_WINDOW_MAX = 10  # maximum K for the final-window mean(sum_p)
R_WINDOW = 10          # window length for first-R / last-R swap counts


# -----------------------------------------------------------------------------
# Param parsing (matches diversity_study / visibility_study conventions)
# -----------------------------------------------------------------------------

def _parse_param(s, n, rng):
    parts = s.split(':')
    mode = parts[0]
    if mode == 'homogeneous':
        return np.full(n, float(parts[1]))
    if mode == 'uniform':
        return rng.uniform(float(parts[1]), float(parts[2]), n)
    raise ValueError(f"Unknown param mode: {mode!r}")


def _agg_log_utility(final_prices):
    """Aggregate household log-utility at termination, U_T = -sum_i log(P_i).

    NaN-safe: drops non-positive prices (defensive).
    """
    if final_prices is None:
        return float('nan')
    P = np.asarray(final_prices, dtype=float)
    P = P[P > 0]
    if P.size == 0:
        return float('nan')
    return float(-np.sum(np.log(P)))


def _build_tier_array(n, tier_mean, tier_std, rng, tier_dist='poisson'):
    """Per-firm tier array.

    Homogeneous branch (tier_std <= 0): constant array = round(tier_mean),
    regardless of tier_dist.

    Heterogeneous branch (tier_std > 0):
      tier_dist='poisson'  : tier_i ~ Poisson(lambda=tier_mean)  (new default)
      tier_dist='lognormal': tier_i ~ round(LogNormal) with target
                              mean = tier_mean, std = tier_std  (legacy)
    """
    if tier_std <= 0:
        return np.full(n, int(round(tier_mean)), dtype=int)
    if tier_mean <= 0:
        return np.zeros(n, dtype=int)
    if tier_dist == 'poisson':
        return rng.poisson(lam=tier_mean, size=n).astype(int)
    if tier_dist == 'lognormal':
        var_n = np.log(1.0 + (tier_std / tier_mean) ** 2)
        mean_n = np.log(tier_mean) - 0.5 * var_n
        draws = rng.lognormal(mean=mean_n, sigma=np.sqrt(var_n), size=n)
        return np.clip(np.round(draws), 0, None).astype(int)
    raise ValueError(f"Unknown tier_dist={tier_dist!r}")


# -----------------------------------------------------------------------------
# Per-trial cost computation
# -----------------------------------------------------------------------------

def cost_metrics_from_result(result):
    """Compute trial-level cost metrics from a simulation result dict.

    sum_p_final_window: mean(sum_p) over the last K = min(10, rounds // 2) rounds,
    where the +1 round at the start is the initial state. Robust to cycling.
    """
    sum_p_hist = result['sum_p_history']     # length rounds + 1, [init, ..., final]
    sum_p_init = float(sum_p_hist[0])
    rounds_done = len(sum_p_hist) - 1
    K = max(1, min(CYCLE_WINDOW_MAX, rounds_done // 2))
    sum_p_final_window = float(sum_p_hist[-K:].mean())
    sum_p_final_last = float(sum_p_hist[-1])
    sum_p_min = float(sum_p_hist.min())

    return {
        'sum_p_init':           sum_p_init,
        'sum_p_final_window':   sum_p_final_window,
        'sum_p_final_last':     sum_p_final_last,
        'sum_p_min':            sum_p_min,
        'cycle_window_K':       K,
    }


# -----------------------------------------------------------------------------
# Static option-B query: per-firm "best-attainable cost given current state"
# -----------------------------------------------------------------------------

def compute_static_gap(base_ns, final_supplier_list, a, b, z,
                       static_max_swaps):
    """For each firm i at the final state, enumerate candidate supplier sets
    reachable by up to `static_max_swaps` simultaneous swaps from its current
    set and pick the candidate with the lowest counterfactual price P[i].

    With `static_max_swaps = min(c, cc)` (the default in main()), the
    enumeration covers EVERY size-c subset of firm i's pool (current
    suppliers + alternates), i.e. "full visibility + unlimited swap" --
    the firm-level cost frontier given the rest of the network's final
    state.  Setting `static_max_swaps = 1` reproduces the original
    ms-reachable-only query.

    For each candidate, the full GE is recomputed with only firm i's
    W-column replaced (other firms' supplier sets fixed at the trial's
    final state).  No firm actually acts; this is a static query.

    Returns:
        p_current   : np.ndarray (n,) -- price under current GE.
        p_best      : np.ndarray (n,) -- min p[i] over enumerated candidates.
        theta_i     : p_current - p_best.
        theta_static: float -- (sum_p_current - sum_p_best) / sum_p_current.
    """
    n = len(final_supplier_list)
    Wbar = base_ns['Wbar']
    AiSi = base_ns['AiSi']
    alt  = base_ns['alternate_supplier_id_list']

    # Current GE (all firms at final supplier sets).
    sup_now = [list(s) for s in final_supplier_list]
    W_now = build_W_from_suppliers(sup_now, Wbar)
    adj_z_now = compute_adjusted_z(AiSi, sup_now, z)
    eq_now = compute_equilibrium_full(a, b, adj_z_now, W_now, n)
    p_current = np.asarray(eq_now['P'])

    # We need to derive each firm's *current* alternates list at the final
    # state.  The base_ns alternates were valid for the INITIAL configuration;
    # any firm that swapped during the dynamics has those swaps moved between
    # supplier_list and alternate_supplier_id_list.  Reconstruct alternates as
    # (initial supplier_list[i] ∪ initial alternate_supplier_id_list[i]) minus
    # the firm's current supplier set, since the simulation only moves
    # entries between the two lists.
    base_pool = [set(base_ns['supplier_id_list'][i]) | set(alt[i])
                 for i in range(n)]
    alt_now   = [sorted(base_pool[i] - set(sup_now[i])) for i in range(n)]

    p_best = p_current.copy()

    for i in range(n):
        current_set = set(sup_now[i])
        alternates = alt_now[i]
        # Walk swap_size = 1..static_max_swaps; the union of all size-k
        # swap-out + size-k swap-in subsets covers every size-c subset of
        # the pool when static_max_swaps >= min(c, cc).
        for swap_size in range(1, static_max_swaps + 1):
            if len(alternates) < swap_size or len(current_set) < swap_size:
                continue
            for new_sups in combinations(alternates, swap_size):
                for old_sups in combinations(current_set, swap_size):
                    cand_set = (current_set - set(old_sups)) | set(new_sups)

                    # W_test: same as W_now but with firm i's column replaced.
                    W_test = W_now.copy()
                    W_test[:, i] = 0.0
                    for s in cand_set:
                        W_test[s, i] = Wbar[s, i]

                    # Adjusted z: only firm i's entry changes (its AiSi key).
                    tmp_sup_i = sorted(int(s) for s in cand_set)
                    tmp_supplier_list = sup_now.copy()
                    tmp_supplier_list[i] = tmp_sup_i
                    adj_z_test = compute_adjusted_z(AiSi, tmp_supplier_list, z)

                    new_eq = compute_equilibrium_full(a, b, adj_z_test,
                                                      W_test, n)
                    cost_i = float(new_eq['P'][i])
                    if cost_i < p_best[i]:
                        p_best[i] = cost_i

    theta_i = p_current - p_best
    sum_pc = float(p_current.sum())
    sum_pb = float(p_best.sum())
    theta_static = (sum_pc - sum_pb) / sum_pc if sum_pc > 0 else 0.0
    return p_current, p_best, theta_i, theta_static


# -----------------------------------------------------------------------------
# Per-firm rewire event aggregation (from trace['rewire_events'])
# -----------------------------------------------------------------------------

def aggregate_per_firm_events(rewire_events, n, rounds_run, cycle_period,
                              R=R_WINDOW):
    """Return three numpy int arrays of length n:
        swaps_first_R   : count of swaps for each firm in rounds [1, R_eff]
        swaps_last_R    : count of swaps for each firm in rounds (rounds_run - R_eff, rounds_run]
        swaps_in_cycle  : count of swaps in the last cycle_period rounds
                          (0 for non-cycled trials)

    R_eff = min(R, rounds_run).
    """
    swaps_first  = np.zeros(n, dtype=int)
    swaps_last   = np.zeros(n, dtype=int)
    swaps_cycle  = np.zeros(n, dtype=int)
    R_eff = max(1, min(R, rounds_run))

    last_lo = max(1, rounds_run - R_eff + 1)
    cycle_lo = (max(1, rounds_run - cycle_period + 1)
                if isinstance(cycle_period, int) and cycle_period >= 2 else None)

    for ev in rewire_events:
        r = int(ev['round'])
        f = int(ev['firm'])
        if 1 <= r <= R_eff:
            swaps_first[f] += 1
        if last_lo <= r <= rounds_run:
            swaps_last[f] += 1
        if cycle_lo is not None and cycle_lo <= r <= rounds_run:
            swaps_cycle[f] += 1
    return swaps_first, swaps_last, swaps_cycle


# -----------------------------------------------------------------------------
# CSV row builders
# -----------------------------------------------------------------------------

TRIAL_COLS = [
    'n', 'cc', 'max_swaps', 'aisi_spread', 'sigma_w',
    'mode', 'tier_mean', 'tier_std', 'tier_dist',
    'a_config', 'b_config', 'z_config',
    'tech_seed', 'init_seed',
    'rounds', 'cycle_period', 'converged', 'total_rewirings',
    'utility_init', 'utility_final',
    'U_T',  # aggregate household log-utility at termination, -sum_i log(P_i)
    'sum_p_init', 'sum_p_final_window', 'sum_p_final_last', 'sum_p_min',
    'cycle_window_K',
    'theta_T_sum',          # (sum_p_init - sum_p_final_window) / sum_p_init
    'theta_T_min',          # (sum_p_init - sum_p_min) / sum_p_init
    'theta_T_util',         # utility_final - utility_init  (= log p_geom_init - log p_geom_final)
    # Option B (static query) -----------------------------------------------
    'unstable_trial',       # 1 if rounds == nb_rounds and not converged and not cycled
    'cycled_trial',         # 1 if cycle_period >= 2
    'sum_p_current',        # sum_i p_current_i  (full GE at final state)
    'sum_p_static_best',    # sum_i min over enumerated candidates of p[i]
    'theta_static',         # (sum_p_current - sum_p_static_best) / sum_p_current
    'R_window',             # window length R used for swaps_first_R / swaps_last_R
    'static_max_swaps',     # # simultaneous swaps used for the static-gap enum
                            # (= min(c, cc) by default => full enumeration of
                            # size-c subsets of the firm's pool)
]

FIRM_COLS = [
    'tech_seed', 'init_seed', 'firm_idx',
    'tier_i', 'p_init_i', 'p_final_i', 'p_min_i',
    'n_swaps_i', 'degree_in_i', 'degree_out_init_i', 'degree_out_final_i',
    # Option B per-firm + rewire-event aggregates ---------------------------
    'p_current_i',          # P[i] in the full-GE at final state
    'p_best_static_i',      # min P[i] over ms-reachable candidates
    'theta_static_i',       # p_current_i - p_best_static_i
    'swaps_first_R',        # swaps in rounds 1..R_window
    'swaps_last_R',         # swaps in last R_window rounds
    'swaps_in_cycle',       # swaps in the last cycle_period rounds (0 if not cycled)
]


def build_trial_row(args, n, tier_arr, tech_seed, init_seed, result, costs,
                    static_metrics=None):
    cycle_p = result.get('cycle_period')
    cycle_p_int = int(cycle_p) if isinstance(cycle_p, int) else None
    unstable = int(
        result['rounds'] >= args.nb_rounds
        and not result['converged']
        and cycle_p_int is None
    )
    cycled = int(cycle_p_int is not None and cycle_p_int >= 2)
    row = {
        'n': n,
        'cc': args.cc,
        'max_swaps': args.max_swaps,
        'aisi_spread': args.aisi_spread,
        'sigma_w': args.sigma_w,
        'mode': args.mode,
        'tier_mean': args.tier_mean if args.mode != 'full' else '',
        'tier_std': args.tier_std if args.mode != 'full' else '',
        'tier_dist': args.tier_dist if args.mode != 'full' else '',
        'a_config': args.a_config,
        'b_config': args.b_config,
        'z_config': args.z_config,
        'tech_seed': tech_seed,
        'init_seed': init_seed,
        'rounds': result['rounds'],
        'cycle_period': result['cycle_period'] if result['cycle_period'] is not None else '',
        'converged': int(result['converged']),
        'total_rewirings': result['total_rewirings'],
        'utility_init': result['initial_utility'],
        'utility_final': result['final_utility'],
        'U_T': _agg_log_utility(result.get('final_prices')),
        'sum_p_init': costs['sum_p_init'],
        'sum_p_final_window': costs['sum_p_final_window'],
        'sum_p_final_last': costs['sum_p_final_last'],
        'sum_p_min': costs['sum_p_min'],
        'cycle_window_K': costs['cycle_window_K'],
        'theta_T_sum': (costs['sum_p_init'] - costs['sum_p_final_window']) / costs['sum_p_init'],
        'theta_T_min': (costs['sum_p_init'] - costs['sum_p_min']) / costs['sum_p_init'],
        'theta_T_util': result['final_utility'] - result['initial_utility'],
        'unstable_trial': unstable,
        'cycled_trial': cycled,
    }
    if static_metrics is not None:
        row['sum_p_current']     = static_metrics['sum_p_current']
        row['sum_p_static_best'] = static_metrics['sum_p_static_best']
        row['theta_static']      = static_metrics['theta_static']
        row['R_window']          = R_WINDOW
        row['static_max_swaps']  = static_metrics['static_max_swaps']
    else:
        row['sum_p_current']     = ''
        row['sum_p_static_best'] = ''
        row['theta_static']      = ''
        row['R_window']          = ''
        row['static_max_swaps']  = ''
    return row


def build_firm_rows(tech_seed, init_seed, result, tier_arr,
                    init_supplier_list, final_supplier_list,
                    p_current=None, p_best=None, theta_i=None,
                    swaps_first=None, swaps_last=None, swaps_cycle=None):
    n = len(result['initial_prices'])
    p_init = np.asarray(result['initial_prices'])
    p_final = np.asarray(result['final_prices'])
    p_min = np.asarray(result['min_prices'])
    swaps = np.asarray(result['per_firm_swaps'])

    # out-degrees from supplier lists (count of clients = times firm appears as supplier)
    d_out_init = np.zeros(n, dtype=int)
    for ss in init_supplier_list:
        for s in ss:
            d_out_init[s] += 1
    d_out_final = np.zeros(n, dtype=int)
    for ss in final_supplier_list:
        for s in ss:
            d_out_final[s] += 1
    # in-degree (= number of suppliers) — fixed by network construction (= c)
    d_in = np.array([len(ss) for ss in final_supplier_list])

    rows = []
    for i in range(n):
        row = {
            'tech_seed': tech_seed,
            'init_seed': init_seed,
            'firm_idx': i,
            'tier_i': int(tier_arr[i]),
            'p_init_i': float(p_init[i]),
            'p_final_i': float(p_final[i]),
            'p_min_i': float(p_min[i]),
            'n_swaps_i': int(swaps[i]),
            'degree_in_i': int(d_in[i]),
            'degree_out_init_i': int(d_out_init[i]),
            'degree_out_final_i': int(d_out_final[i]),
        }
        if p_current is not None:
            row['p_current_i']     = float(p_current[i])
            row['p_best_static_i'] = float(p_best[i])
            row['theta_static_i']  = float(theta_i[i])
        else:
            row['p_current_i']     = ''
            row['p_best_static_i'] = ''
            row['theta_static_i']  = ''
        if swaps_first is not None:
            row['swaps_first_R']  = int(swaps_first[i])
            row['swaps_last_R']   = int(swaps_last[i])
            row['swaps_in_cycle'] = int(swaps_cycle[i])
        else:
            row['swaps_first_R']  = ''
            row['swaps_last_R']   = ''
            row['swaps_in_cycle'] = ''
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Main loop
# -----------------------------------------------------------------------------

def existing_keys(path, key_cols):
    """Return set of (tuple of key values) already present in the CSV."""
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add(tuple(row[k] for k in key_cols))
    return keys


def open_csv_writer(path, fieldnames):
    is_new = not os.path.exists(path)
    f = open(path, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if is_new:
        writer.writeheader()
    return f, writer


def main():
    p = argparse.ArgumentParser(description='Cost-reduction (theta_T) study')
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--c', type=int, default=4)
    p.add_argument('--cc', type=int, default=4)
    p.add_argument('--max_swaps', type=int, default=1,
                   help='Max simultaneous swaps allowed to firms DURING the '
                        'dynamics (typical: 1).')
    p.add_argument('--static_max_swaps', type=int, default=None,
                   help='Max simultaneous swaps used by the post-hoc static-gap '
                        '(option-B) query. Default: min(c, cc) -- i.e. '
                        'enumerate every size-c subset of the firm\'s pool '
                        '("full visibility + unlimited swap"). Pass '
                        '--static_max_swaps 1 to recover the original '
                        'ms-reachable-only behaviour.')
    p.add_argument('--aisi_spread', type=float, default=0.0)
    p.add_argument('--sigma_w', type=float, default=0.0)
    p.add_argument('--a_config', default='homogeneous:0.5')
    p.add_argument('--b_config', default='homogeneous:0.9')
    p.add_argument('--z_config', default='homogeneous:1.0')
    p.add_argument('--mode', choices=['limited', 'full'], default='limited')
    p.add_argument('--tier_mean', type=float, default=2.0,
                   help='Mean tier visibility (ignored when mode=full). For '
                        'tier_dist=poisson this is lambda.')
    p.add_argument('--tier_std', type=float, default=0.0,
                   help='Std tier visibility. 0 = homogeneous; >0 = hetero. '
                        'Ignored numerically for tier_dist=poisson but kept '
                        'as the hetero flag.')
    p.add_argument('--tier_dist', type=str, default='poisson',
                   choices=['poisson', 'lognormal'],
                   help='Hetero-tau distribution (default: poisson; legacy: '
                        'lognormal).')
    p.add_argument('--tech_per_job', type=int, default=5)
    p.add_argument('--inits_per_tech', type=int, default=30,
                   help='Random initial networks per tech matrix.')
    p.add_argument('--nb_rounds', type=int, default=200)
    p.add_argument('--base_seed', type=int, default=0)
    p.add_argument('--output', default=None,
                   help='trial-level CSV path. firm-level CSV uses '
                        '<output_stem>_firm.csv.')
    args = p.parse_args()

    if args.output is None:
        out_dir = os.path.join(REPO_ROOT, 'results', 'cost')
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(
            out_dir,
            f'cost_n{args.n}_mode{args.mode}_taum{args.tier_mean}_'
            f'taus{args.tier_std}_seed{args.base_seed}.csv'
        )

    trial_path = args.output
    stem, ext = os.path.splitext(trial_path)
    firm_path = stem + '_firm' + ext

    os.makedirs(os.path.dirname(trial_path) or '.', exist_ok=True)

    # Resolve the static-gap enumeration budget.  Default = min(c, cc), which
    # makes the option-B query cover every size-c subset of each firm's pool
    # ("full visibility + unlimited swap").
    static_max_swaps = (args.static_max_swaps
                        if args.static_max_swaps is not None
                        else min(args.c, args.cc))

    # Resume keys (per trial)
    resume_keys = existing_keys(trial_path, ['tech_seed', 'init_seed'])

    f_trial, w_trial = open_csv_writer(trial_path, TRIAL_COLS)
    f_firm,  w_firm  = open_csv_writer(firm_path,  FIRM_COLS)

    rng_master = np.random.default_rng(args.base_seed)
    n = args.n

    print(f"Cost-reduction study | n={n} mode={args.mode} "
          f"tau_mean={args.tier_mean} tau_std={args.tier_std} | "
          f"tech_per_job={args.tech_per_job} inits_per_tech={args.inits_per_tech}")
    print(f"  trial CSV: {trial_path}")
    print(f"  firm  CSV: {firm_path}")

    # Outer: tech matrices. Seed budgeting (stays within 2**32):
    #   tech_seed = base_seed * 10_000 + tech_idx       (< ~5e8 if base_seed < 1e5)
    #   init_seed = tech_seed * 100   + init_idx        (< ~5e10... too big; reduce)
    # Use modular arithmetic so all seeds stay below 2**31.
    SEED_MOD = 2**31 - 1
    n_done = 0
    for tech_idx in range(args.tech_per_job):
        tech_seed = (args.base_seed * 10_000 + tech_idx) % SEED_MOD
        rng_tech = np.random.default_rng(tech_seed)

        a_arr = _parse_param(args.a_config, n, rng_tech)
        b_arr = _parse_param(args.b_config, n, rng_tech)
        z_arr = _parse_param(args.z_config, n, rng_tech)

        base_ns = generate_base_network(
            n=n, c=args.c, cc=args.cc, aisi_spread=args.aisi_spread,
            seed=tech_seed, a=a_arr, b=b_arr, sigma_w=args.sigma_w,
        )

        for init_idx in range(args.inits_per_tech):
            init_seed = (tech_seed * 100 + init_idx) % SEED_MOD
            key = (str(tech_seed), str(init_seed))
            if key in resume_keys:
                continue

            init_ns = generate_random_initial_network(
                n=n, Wbar=base_ns['Wbar'], AiSi=base_ns['AiSi'], seed=int(init_seed),
            )
            init_supplier_list = [list(s) for s in init_ns['supplier_id_list']]

            # Tier array: hetero is per (tech_seed, init_seed) — re-drawn per init for
            # cleaner Monte Carlo (otherwise the same tier_i pattern persists across
            # trials sharing a tech matrix). Set seed for reproducibility.
            tier_rng = np.random.default_rng((init_seed + 7919) % SEED_MOD)
            tier_arr = _build_tier_array(n, args.tier_mean, args.tier_std,
                                          tier_rng, tier_dist=args.tier_dist)

            kwargs = dict(
                a=a_arr, b=b_arr, z=z_arr,
                mode=args.mode, max_swaps=args.max_swaps,
                nb_rounds=args.nb_rounds, seed=int((init_seed + 31337) % SEED_MOD),
                trace=True,
            )
            if args.mode == 'limited':
                kwargs['tier'] = tier_arr

            result = run_unified_simulation(init_ns, **kwargs)

            costs = cost_metrics_from_result(result)

            # ---- Option B: static "best-attainable cost given current state" --
            # Now uses static_max_swaps (default = min(c, cc)) so the
            # enumeration covers all size-c subsets of the firm's pool
            # under full-visibility evaluation -- decoupled from the
            # per-round budget args.max_swaps used by the dynamics.
            p_current, p_best, theta_i, theta_static_val = compute_static_gap(
                base_ns, result['final_supplier_list'],
                a_arr, b_arr, z_arr, static_max_swaps,
            )
            static_metrics = {
                'sum_p_current':     float(p_current.sum()),
                'sum_p_static_best': float(p_best.sum()),
                'theta_static':      float(theta_static_val),
                'static_max_swaps':  int(static_max_swaps),
            }

            # ---- Per-firm rewire-event windows ---------------------------------
            trace = result.get('trace', {}) or {}
            rewire_events = trace.get('rewire_events', []) or []
            cycle_p = result.get('cycle_period')
            cycle_p_int = int(cycle_p) if isinstance(cycle_p, int) else None
            swaps_first, swaps_last, swaps_cycle = aggregate_per_firm_events(
                rewire_events, n, result['rounds'], cycle_p_int,
            )

            trow = build_trial_row(args, n, tier_arr, tech_seed, init_seed,
                                   result, costs, static_metrics=static_metrics)
            w_trial.writerow(trow)

            for frow in build_firm_rows(
                tech_seed, init_seed, result, tier_arr,
                init_supplier_list, result['final_supplier_list'],
                p_current=p_current, p_best=p_best, theta_i=theta_i,
                swaps_first=swaps_first, swaps_last=swaps_last,
                swaps_cycle=swaps_cycle,
            ):
                w_firm.writerow(frow)

            n_done += 1
            if n_done % 25 == 0:
                f_trial.flush()
                f_firm.flush()
                print(f"  ... {n_done} trials done")

    f_trial.close()
    f_firm.close()
    print(f"Done. {n_done} trials written.")


if __name__ == '__main__':
    main()
