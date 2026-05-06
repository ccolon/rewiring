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

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.networks import generate_base_network, generate_random_initial_network
from rewiring.simulation import run_unified_simulation


CYCLE_WINDOW_MAX = 10  # maximum K for the final-window mean(sum_p)


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


def _build_tier_array(n, tier_mean, tier_std, rng):
    """Per-firm tier array. tier_std=0 -> homogeneous; >0 -> lognormal draw
    with target mean tier_mean and target std tier_std (matches
    visibility_study.build_tier_array).
    """
    if tier_std <= 0:
        return np.full(n, int(round(tier_mean)), dtype=int)
    if tier_mean <= 0:
        return np.zeros(n, dtype=int)
    var_n = np.log(1.0 + (tier_std / tier_mean) ** 2)
    mean_n = np.log(tier_mean) - 0.5 * var_n
    draws = rng.lognormal(mean=mean_n, sigma=np.sqrt(var_n), size=n)
    return np.clip(np.round(draws), 0, None).astype(int)


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
# CSV row builders
# -----------------------------------------------------------------------------

TRIAL_COLS = [
    'n', 'cc', 'max_swaps', 'aisi_spread', 'sigma_w',
    'mode', 'tier_mean', 'tier_std',
    'a_config', 'b_config', 'z_config',
    'tech_seed', 'init_seed',
    'rounds', 'cycle_period', 'converged', 'total_rewirings',
    'utility_init', 'utility_final',
    'sum_p_init', 'sum_p_final_window', 'sum_p_final_last', 'sum_p_min',
    'cycle_window_K',
    'theta_T_sum',          # (sum_p_init - sum_p_final_window) / sum_p_init
    'theta_T_min',          # (sum_p_init - sum_p_min) / sum_p_init
    'theta_T_util',         # utility_final - utility_init  (= log p_geom_init - log p_geom_final)
]

FIRM_COLS = [
    'tech_seed', 'init_seed', 'firm_idx',
    'tier_i', 'p_init_i', 'p_final_i', 'p_min_i',
    'n_swaps_i', 'degree_in_i', 'degree_out_init_i', 'degree_out_final_i',
]


def build_trial_row(args, n, tier_arr, tech_seed, init_seed, result, costs):
    return {
        'n': n,
        'cc': args.cc,
        'max_swaps': args.max_swaps,
        'aisi_spread': args.aisi_spread,
        'sigma_w': args.sigma_w,
        'mode': args.mode,
        'tier_mean': args.tier_mean if args.mode != 'full' else '',
        'tier_std': args.tier_std if args.mode != 'full' else '',
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
        'sum_p_init': costs['sum_p_init'],
        'sum_p_final_window': costs['sum_p_final_window'],
        'sum_p_final_last': costs['sum_p_final_last'],
        'sum_p_min': costs['sum_p_min'],
        'cycle_window_K': costs['cycle_window_K'],
        'theta_T_sum': (costs['sum_p_init'] - costs['sum_p_final_window']) / costs['sum_p_init'],
        'theta_T_min': (costs['sum_p_init'] - costs['sum_p_min']) / costs['sum_p_init'],
        'theta_T_util': result['final_utility'] - result['initial_utility'],
    }


def build_firm_rows(tech_seed, init_seed, result, tier_arr, init_supplier_list, final_supplier_list):
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
        rows.append({
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
        })
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
    p.add_argument('--max_swaps', type=int, default=1)
    p.add_argument('--aisi_spread', type=float, default=0.0)
    p.add_argument('--sigma_w', type=float, default=0.0)
    p.add_argument('--a_config', default='homogeneous:0.5')
    p.add_argument('--b_config', default='homogeneous:0.9')
    p.add_argument('--z_config', default='homogeneous:1.0')
    p.add_argument('--mode', choices=['limited', 'full'], default='limited')
    p.add_argument('--tier_mean', type=float, default=2.0,
                   help='Mean tier visibility (ignored when mode=full).')
    p.add_argument('--tier_std', type=float, default=0.0,
                   help='Std tier visibility. 0 = homogeneous; >0 = lognormal hetero draw.')
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
            tier_arr = _build_tier_array(n, args.tier_mean, args.tier_std, tier_rng)

            kwargs = dict(
                a=a_arr, b=b_arr, z=z_arr,
                mode=args.mode, max_swaps=args.max_swaps,
                nb_rounds=args.nb_rounds, seed=int((init_seed + 31337) % SEED_MOD),
            )
            if args.mode == 'limited':
                kwargs['tier'] = tier_arr

            result = run_unified_simulation(init_ns, **kwargs)

            costs = cost_metrics_from_result(result)
            trow = build_trial_row(args, n, tier_arr, tech_seed, init_seed, result, costs)
            w_trial.writerow(trow)

            for frow in build_firm_rows(tech_seed, init_seed, result, tier_arr,
                                         init_supplier_list, result['final_supplier_list']):
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
