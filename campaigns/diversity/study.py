"""
Diversity study: measures diversity of final equilibria as a function of network size.

For each n in [N_MIN, N_MAX]:
  - Generate N_TECH_MATRICES technology matrices (Wbar + AiSi)
  - For each, run N_TRIALS full-anticipation simulations in two conditions:
      Blue: same initial network, different permutation orders
            -> measures sensitivity to evaluation order alone
      Red:  different initial networks, different permutation orders
            -> measures path-dependence (uniqueness of equilibrium)
  - Diversity = (# unique final configurations - 1) / (N_TRIALS - 1)
      0%   all trials reach the same final network
      100% every trial reaches a different final network

Results are saved to CSV incrementally (supports resume on interruption).
All parameters are recorded in every row so runs with different settings
can safely be appended to the same file or kept in separate files.

RESUME NOTE: the resume key is (n, tech_idx, series). If you change
parameters and reuse the same OUTPUT_FILE, already-computed rows will be
skipped even though they used different parameters. Use a distinct
OUTPUT_FILE for each parameter combination to avoid this.
"""

import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

# Allow `python campaigns/diversity/study.py` from anywhere. This script lives
# 2 levels below the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir)))

from rewiring.networks import generate_base_network, generate_random_initial_network
from rewiring.parameters import generate_a_parameter, generate_parameter
from rewiring.simulation import run_unified_simulation


# =============================================================================
# CONFIGURATION  (defaults; overridden by CLI args when provided)
# =============================================================================

N_MIN = 10
N_MAX = 50
N_TECH_MATRICES = 50
N_TRIALS = 50
NB_ROUNDS = 20

C = 4
CC = 4
AISI_SPREAD = 0.1
SIGMA_W = 0.0
MAX_SWAPS = 1

A_CONFIG = {'mode': 'homogeneous', 'value': 0.5}
B_CONFIG = {'mode': 'homogeneous', 'value': 0.9}
Z_CONFIG = {'mode': 'homogeneous', 'value': 1.0}

# Anticipation mode: "full" (default; original n=100 main sweep) or "limited"
# (boundary-conditioned partial-equilibrium evaluation, requires tier params).
MODE = "full"
TIER_MEAN = 0.0    # used only when MODE == "limited"; when TIER_STD == 0 the
TIER_STD  = 0.0    # tier_arr is constant = round(TIER_MEAN); else drawn from
                   # TIER_DIST (default: poisson).
TIER_DIST = 'poisson'  # 'poisson' (default) or 'lognormal' (legacy).

BASE_SEED = 0  # Monte Carlo offset; different values -> independent RNG streams

# Which diversity series to compute: 'both' | 'same_init_only' | 'dif_init_only'.
# 'dif_init_only' halves runtime when only same_tech_dif_init is needed.
SERIES_FILTER = 'both'

OUTPUT_FILE = "diversity_results.csv"


def _build_tier_array(n, tier_mean, tier_std, rng, tier_dist='poisson'):
    """Per-firm tier array.

    Homogeneous (tier_std <= 0): constant array = round(tier_mean).
    Heterogeneous (tier_std > 0):
      tier_dist='poisson'  : tier_i ~ Poisson(lambda=tier_mean)  (default)
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


# =============================================================================
# DIVERSITY METRIC
# =============================================================================

def compute_diversity(final_supplier_lists):
    """
    Diversity = (# unique final configurations - 1) / (K - 1).

    Each configuration is the frozenset of (firm, supplier) edges in the
    final network.  Returns 0.0 when K <= 1.
    """
    k = len(final_supplier_lists)
    if k <= 1:
        return 0.0
    configs = set()
    for supplier_list in final_supplier_lists:
        config = frozenset(
            (firm, sup)
            for firm, suppliers in enumerate(supplier_list)
            for sup in suppliers
        )
        configs.add(config)
    return (len(configs) - 1) / (k - 1)


# =============================================================================
# CSV HELPERS
# =============================================================================

CSV_FIELDNAMES = [
    'n', 'tech_idx', 'series', 'diversity',
    # Convergence diagnostics. A trial can exit one of three ways:
    #   - rewirings == 0 in a full round   -> period-1, counted in `frac_converged`
    #   - period-k limit cycle (k in 2..MAX_CYCLE_PERIOD) -> counted in `frac_cycled`
    #   - hit nb_rounds without either     -> truncated; not in either fraction
    # `mean_cycle_period` is the average detected period over the cycling trials
    # (NaN if no trial cycled).
    # Trustworthy cells satisfy
    #     frac_converged + frac_cycled >= 1 - epsilon
    # and `max_rounds < nb_rounds`.
    'frac_converged', 'frac_cycled', 'mean_cycle_period',
    'mean_rounds', 'max_rounds',
    # Mean / max accepted swaps per firm across the n_trials trials.
    # At ms=1 this is a count; at higher ms it's the simultaneous-swap-event count
    # divided by n. Useful for "how much rewiring happened before stabilization".
    'mean_swaps_per_firm', 'max_swaps_per_firm',
    'c', 'cc', 'aisi_spread', 'sigma_w', 'max_swaps', 'nb_rounds', 'n_trials',
    'a_config', 'b_config', 'z_config',
    # Anticipation mode and per-firm tier-visibility config (used only when
    # mode == "limited"). tier_mean and tier_std are the input distribution
    # parameters; each tech matrix's actual realised tier_arr is determined
    # by the tech_seed.
    'mode', 'tier_mean', 'tier_std', 'tier_dist',
    'U_T',  # aggregate household log-utility, -sum_i log(P_i),
            # averaged across trials in this cell
    'base_seed', 'tech_seed',
]


def load_existing_keys(filepath):
    """Return the set of (n, tech_idx, series) rows already in the CSV."""
    keys = set()
    if not os.path.exists(filepath):
        return keys
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((int(row['n']), int(row['tech_idx']), row['series']))
    return keys


def append_row(filepath, row_dict):
    """Append a single row to the CSV, writing the header if the file is new."""
    write_header = not os.path.exists(filepath)
    with open(filepath, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


# =============================================================================
# MAIN STUDY LOOP
# =============================================================================

def run_study():
    existing_keys = load_existing_keys(OUTPUT_FILE)
    n_values = list(range(N_MIN, N_MAX + 1))
    total = len(n_values) * N_TECH_MATRICES * 2
    done = len(existing_keys)

    print(f"Diversity study  n=[{N_MIN},{N_MAX}]  {N_TECH_MATRICES} tech matrices  {N_TRIALS} trials each")
    print(f"  Network : c={C}, cc={CC}, aisi_spread={AISI_SPREAD}, sigma_w={SIGMA_W}")
    print(f"  Economic: a={A_CONFIG}, b={B_CONFIG}, z={Z_CONFIG}")
    print(f"  Sim     : nb_rounds={NB_ROUNDS}, max_swaps={MAX_SWAPS}, mode={MODE}, "
          f"tier_mean={TIER_MEAN}, tier_std={TIER_STD}")
    print(f"  Output  : {OUTPUT_FILE}  (resuming {done}/{total} already done)")
    print("=" * 70)

    for n in n_values:
        for tech_idx in range(N_TECH_MATRICES):

            blue_key = (n, tech_idx, 'same_tech_same_init')
            red_key  = (n, tech_idx, 'same_tech_dif_init')

            if blue_key in existing_keys and red_key in existing_keys:
                done += 2
                continue

            # ------------------------------------------------------------------
            # Build technology matrix for this (n, tech_idx) pair.
            # tech_seed = BASE_SEED * 10000 + tech_idx so different BASE_SEED
            # values give independent Monte Carlo streams.
            # ------------------------------------------------------------------
            tech_seed = BASE_SEED * 10000 + tech_idx
            seed_off = BASE_SEED * 10000

            # Draw economic parameters before generate_base_network re-seeds.
            random.seed(tech_seed)
            np.random.seed(tech_seed)
            b = generate_parameter(B_CONFIG, n, param_name='b', verbose=False)
            a = generate_a_parameter(A_CONFIG, b, n, verbose=False)
            z = generate_parameter(Z_CONFIG, n, param_name='z', verbose=False)

            # generate_base_network re-seeds internally with tech_seed.
            base_state = generate_base_network(
                n, C, CC, AISI_SPREAD, seed=tech_seed,
                a=a, b=b, sigma_w=SIGMA_W,
            )
            Wbar = base_state['Wbar']
            AiSi = base_state['AiSi']

            # Per-tech tier-visibility array (only used when MODE == "limited").
            # Build a fresh np.random.Generator from tech_seed to avoid disturbing
            # the global numpy RNG state used by perm_seed inside the simulator.
            tier_arr = None
            if MODE in ("limited", "naive_limited"):
                tier_rng = np.random.default_rng(tech_seed + 7919)  # +prime: avoid collision
                tier_arr = _build_tier_array(n, TIER_MEAN, TIER_STD, tier_rng,
                                              tier_dist=TIER_DIST)

            common = {
                'n': n, 'tech_idx': tech_idx,
                'c': C, 'cc': CC, 'aisi_spread': AISI_SPREAD, 'sigma_w': SIGMA_W,
                'max_swaps': MAX_SWAPS, 'nb_rounds': NB_ROUNDS, 'n_trials': N_TRIALS,
                'a_config': json.dumps(A_CONFIG),
                'b_config': json.dumps(B_CONFIG),
                'z_config': json.dumps(Z_CONFIG),
                'mode': MODE,
                'tier_mean': TIER_MEAN,
                'tier_std': TIER_STD,
                'tier_dist': TIER_DIST if MODE in ("limited", "naive_limited") else '',
                'base_seed': BASE_SEED,
                'tech_seed': tech_seed,
            }

            # ------------------------------------------------------------------
            # same_tech_same_init — same initial network, different permutation orders
            #   perm_seed = 1000 + trial  controls np.random.permutation inside
            #   the simulation (evaluation order of firms each round).
            # ------------------------------------------------------------------
            if blue_key not in existing_keys and SERIES_FILTER != 'dif_init_only':
                t0 = time.time()
                final_lists, rounds_list, conv_list, cycle_periods_list, rewires_list, U_T_list = [], [], [], [], [], []
                for trial in range(N_TRIALS):
                    result = run_unified_simulation(
                        base_state, a, b, z, mode=MODE,
                        seed=seed_off + 1000 + trial,
                        max_swaps=MAX_SWAPS,
                        nb_rounds=NB_ROUNDS,
                        tier=tier_arr,
                    )
                    final_lists.append(result['final_supplier_list'])
                    rounds_list.append(int(result['rounds']))
                    conv_list.append(bool(result['converged']))
                    cycle_periods_list.append(result.get('cycle_period'))
                    rewires_list.append(int(result.get('total_rewirings', 0)))
                    P_T = np.asarray(result.get('final_prices', []), dtype=float)
                    P_T = P_T[P_T > 0]
                    U_T_list.append(float(-np.sum(np.log(P_T))) if P_T.size else float('nan'))

                diversity = compute_diversity(final_lists)
                rewires_arr = np.asarray(rewires_list, dtype=float) / float(n)
                cycled = [cp for cp in cycle_periods_list if isinstance(cp, int) and cp >= 2]
                row = {**common, 'series': 'same_tech_same_init', 'diversity': diversity,
                       'frac_converged':       float(np.mean(conv_list)),
                       'frac_cycled':          float(len(cycled)) / max(N_TRIALS, 1),
                       'mean_cycle_period':    (float(np.mean(cycled)) if cycled else float('nan')),
                       'mean_rounds':          float(np.mean(rounds_list)),
                       'max_rounds':           int(np.max(rounds_list)),
                       'mean_swaps_per_firm':  float(np.mean(rewires_arr)),
                       'max_swaps_per_firm':   float(np.max(rewires_arr)),
                       'U_T':                  float(np.nanmean(U_T_list))}
                append_row(OUTPUT_FILE, row)
                done += 1
                elapsed = time.time() - t0
                print(f"[{done:5d}/{total}] n={n:2d} tech={tech_idx:2d} same_tech_same_init  "
                      f"diversity={diversity:.3f}  conv={row['frac_converged']:.2f}  "
                      f"cyc={row['frac_cycled']:.2f}(<k>={row['mean_cycle_period']:.1f})  "
                      f"rounds={row['mean_rounds']:.1f}/{row['max_rounds']}  "
                      f"swaps/firm={row['mean_swaps_per_firm']:.2f}  ({elapsed:.1f}s)")

            # ------------------------------------------------------------------
            # same_tech_dif_init — different initial networks, different permutation orders
            #   init_seed = 2000 + trial  controls the random initial supplier
            #   selection via generate_random_initial_network.
            #   perm_seed = 3000 + trial  controls firm evaluation order.
            # ------------------------------------------------------------------
            if red_key not in existing_keys and SERIES_FILTER != 'same_init_only':
                t0 = time.time()
                final_lists, rounds_list, conv_list, cycle_periods_list, rewires_list, U_T_list = [], [], [], [], [], []
                for trial in range(N_TRIALS):
                    trial_state = generate_random_initial_network(
                        n, Wbar, AiSi, seed=seed_off + 2000 + trial,
                    )
                    result = run_unified_simulation(
                        trial_state, a, b, z, mode=MODE,
                        seed=seed_off + 3000 + trial,
                        max_swaps=MAX_SWAPS,
                        nb_rounds=NB_ROUNDS,
                        tier=tier_arr,
                    )
                    final_lists.append(result['final_supplier_list'])
                    rounds_list.append(int(result['rounds']))
                    conv_list.append(bool(result['converged']))
                    cycle_periods_list.append(result.get('cycle_period'))
                    rewires_list.append(int(result.get('total_rewirings', 0)))
                    P_T = np.asarray(result.get('final_prices', []), dtype=float)
                    P_T = P_T[P_T > 0]
                    U_T_list.append(float(-np.sum(np.log(P_T))) if P_T.size else float('nan'))

                diversity = compute_diversity(final_lists)
                rewires_arr = np.asarray(rewires_list, dtype=float) / float(n)
                cycled = [cp for cp in cycle_periods_list if isinstance(cp, int) and cp >= 2]
                row = {**common, 'series': 'same_tech_dif_init', 'diversity': diversity,
                       'frac_converged':       float(np.mean(conv_list)),
                       'frac_cycled':          float(len(cycled)) / max(N_TRIALS, 1),
                       'mean_cycle_period':    (float(np.mean(cycled)) if cycled else float('nan')),
                       'mean_rounds':          float(np.mean(rounds_list)),
                       'max_rounds':           int(np.max(rounds_list)),
                       'mean_swaps_per_firm':  float(np.mean(rewires_arr)),
                       'max_swaps_per_firm':   float(np.max(rewires_arr)),
                       'U_T':                  float(np.nanmean(U_T_list))}
                append_row(OUTPUT_FILE, row)
                done += 1
                elapsed = time.time() - t0
                print(f"[{done:5d}/{total}] n={n:2d} tech={tech_idx:2d} same_tech_dif_init  "
                      f"diversity={diversity:.3f}  conv={row['frac_converged']:.2f}  "
                      f"cyc={row['frac_cycled']:.2f}(<k>={row['mean_cycle_period']:.1f})  "
                      f"rounds={row['mean_rounds']:.1f}/{row['max_rounds']}  "
                      f"swaps/firm={row['mean_swaps_per_firm']:.2f}  ({elapsed:.1f}s)")

    print("=" * 70)
    print(f"Done. Results saved to {OUTPUT_FILE}")


def parse_config_arg(s):
    """Parse a config string like 'homogeneous:0.5' or 'uniform:0.5:1.5' into a dict."""
    parts = s.split(':')
    mode = parts[0]
    if mode == 'homogeneous':
        return {'mode': 'homogeneous', 'value': float(parts[1])}
    elif mode == 'uniform':
        return {'mode': 'uniform', 'min': float(parts[1]), 'max': float(parts[2])}
    else:
        raise ValueError(f"Unknown config mode: {mode}")


def parse_args():
    parser = argparse.ArgumentParser(description='Diversity study')
    parser.add_argument('--n_min', type=int, default=None)
    parser.add_argument('--n_max', type=int, default=None)
    parser.add_argument('--n_tech', type=int, default=None)
    parser.add_argument('--n_trials', type=int, default=None)
    parser.add_argument('--nb_rounds', type=int, default=None)
    parser.add_argument('--cc', type=int, default=None)
    parser.add_argument('--aisi_spread', type=float, default=None)
    parser.add_argument('--sigma_w', type=float, default=None)
    parser.add_argument('--max_swaps', type=int, default=None)
    parser.add_argument('--b_config', type=str, default=None,
                        help='e.g. homogeneous:0.9 or uniform:0.5:1.5')
    parser.add_argument('--a_config', type=str, default=None,
                        help='e.g. homogeneous:0.5 or uniform:0.3:0.7')
    parser.add_argument('--z_config', type=str, default=None,
                        help='e.g. homogeneous:1.0 or uniform:0.5:2.0')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['full', 'limited', 'naive_limited', 'aa',
                                 'full_profitmax'],
                        help='Anticipation/objective mode: full (default), '
                             'limited (boundary-conditioned partial GE; needs '
                             '--tier_mean), naive_limited, aa, or '
                             "full_profitmax (appendix 'Profit maximisation' "
                             '-- DRS only).')
    parser.add_argument('--tier_mean', type=float, default=None,
                        help='Mean tier visibility (used when mode=limited/naive_limited).')
    parser.add_argument('--tier_std', type=float, default=None,
                        help='Std tier visibility. 0 = homogeneous; >0 = '
                             'heterogeneous draw per tech matrix from --tier_dist.')
    parser.add_argument('--tier_dist', type=str, default=None,
                        choices=['poisson', 'lognormal'],
                        help='Hetero-tau distribution (default: poisson; '
                             'legacy: lognormal).')
    parser.add_argument('--base_seed', type=int, default=None,
                        help='Monte Carlo offset; different values give independent RNG streams')
    parser.add_argument('--series_filter', type=str, default=None,
                        choices=['both', 'same_init_only', 'dif_init_only'],
                        help='Which series to compute. dif_init_only halves runtime '
                             'when only same_tech_dif_init is needed.')
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.n_min is not None:
        N_MIN = args.n_min
    if args.n_max is not None:
        N_MAX = args.n_max
    if args.n_tech is not None:
        N_TECH_MATRICES = args.n_tech
    if args.n_trials is not None:
        N_TRIALS = args.n_trials
    if args.nb_rounds is not None:
        NB_ROUNDS = args.nb_rounds
    if args.cc is not None:
        CC = args.cc
    if args.aisi_spread is not None:
        AISI_SPREAD = args.aisi_spread
    if args.sigma_w is not None:
        SIGMA_W = args.sigma_w
    if args.max_swaps is not None:
        MAX_SWAPS = args.max_swaps
    if args.b_config is not None:
        B_CONFIG = parse_config_arg(args.b_config)
    if args.a_config is not None:
        A_CONFIG = parse_config_arg(args.a_config)
    if args.z_config is not None:
        Z_CONFIG = parse_config_arg(args.z_config)
    if args.mode is not None:
        MODE = args.mode
    if args.tier_mean is not None:
        TIER_MEAN = args.tier_mean
    if args.tier_std is not None:
        TIER_STD = args.tier_std
    if args.tier_dist is not None:
        TIER_DIST = args.tier_dist
    if args.base_seed is not None:
        BASE_SEED = args.base_seed
    if args.series_filter is not None:
        SERIES_FILTER = args.series_filter
    if args.output is not None:
        OUTPUT_FILE = args.output

    if MODE in ("limited", "naive_limited") and args.tier_mean is None:
        raise SystemExit(f"--mode={MODE} requires --tier_mean (and optional --tier_std).")

    run_study()
