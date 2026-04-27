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

# Allow `python scripts/diversity_study.py` from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

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

BASE_SEED = 0  # Monte Carlo offset; different values -> independent RNG streams

OUTPUT_FILE = "diversity_results.csv"


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
    # Convergence diagnostics: trials may hit nb_rounds without reaching a fixed
    # point. frac_converged < 1 or max_rounds == nb_rounds flags potentially
    # transient (truncated) runs whose `final_supplier_list` is not a true
    # equilibrium and may inflate `diversity`.
    'frac_converged', 'mean_rounds', 'max_rounds',
    'c', 'cc', 'aisi_spread', 'sigma_w', 'max_swaps', 'nb_rounds', 'n_trials',
    'a_config', 'b_config', 'z_config',
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
    print(f"  Sim     : nb_rounds={NB_ROUNDS}, max_swaps={MAX_SWAPS}")
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

            common = {
                'n': n, 'tech_idx': tech_idx,
                'c': C, 'cc': CC, 'aisi_spread': AISI_SPREAD, 'sigma_w': SIGMA_W,
                'max_swaps': MAX_SWAPS, 'nb_rounds': NB_ROUNDS, 'n_trials': N_TRIALS,
                'a_config': json.dumps(A_CONFIG),
                'b_config': json.dumps(B_CONFIG),
                'z_config': json.dumps(Z_CONFIG),
                'base_seed': BASE_SEED,
                'tech_seed': tech_seed,
            }

            # ------------------------------------------------------------------
            # same_tech_same_init — same initial network, different permutation orders
            #   perm_seed = 1000 + trial  controls np.random.permutation inside
            #   the simulation (evaluation order of firms each round).
            # ------------------------------------------------------------------
            if blue_key not in existing_keys:
                t0 = time.time()
                final_lists, rounds_list, conv_list = [], [], []
                for trial in range(N_TRIALS):
                    result = run_unified_simulation(
                        base_state, a, b, z, mode="full",
                        seed=seed_off + 1000 + trial,
                        max_swaps=MAX_SWAPS,
                        nb_rounds=NB_ROUNDS,
                    )
                    final_lists.append(result['final_supplier_list'])
                    rounds_list.append(int(result['rounds']))
                    conv_list.append(bool(result['converged']))

                diversity = compute_diversity(final_lists)
                row = {**common, 'series': 'same_tech_same_init', 'diversity': diversity,
                       'frac_converged': float(np.mean(conv_list)),
                       'mean_rounds':    float(np.mean(rounds_list)),
                       'max_rounds':     int(np.max(rounds_list))}
                append_row(OUTPUT_FILE, row)
                done += 1
                elapsed = time.time() - t0
                print(f"[{done:5d}/{total}] n={n:2d} tech={tech_idx:2d} same_tech_same_init  "
                      f"diversity={diversity:.3f}  conv={row['frac_converged']:.2f}  "
                      f"rounds={row['mean_rounds']:.1f}/{row['max_rounds']}  ({elapsed:.1f}s)")

            # ------------------------------------------------------------------
            # same_tech_dif_init — different initial networks, different permutation orders
            #   init_seed = 2000 + trial  controls the random initial supplier
            #   selection via generate_random_initial_network.
            #   perm_seed = 3000 + trial  controls firm evaluation order.
            # ------------------------------------------------------------------
            if red_key not in existing_keys:
                t0 = time.time()
                final_lists, rounds_list, conv_list = [], [], []
                for trial in range(N_TRIALS):
                    trial_state = generate_random_initial_network(
                        n, Wbar, AiSi, seed=seed_off + 2000 + trial,
                    )
                    result = run_unified_simulation(
                        trial_state, a, b, z, mode="full",
                        seed=seed_off + 3000 + trial,
                        max_swaps=MAX_SWAPS,
                        nb_rounds=NB_ROUNDS,
                    )
                    final_lists.append(result['final_supplier_list'])
                    rounds_list.append(int(result['rounds']))
                    conv_list.append(bool(result['converged']))

                diversity = compute_diversity(final_lists)
                row = {**common, 'series': 'same_tech_dif_init', 'diversity': diversity,
                       'frac_converged': float(np.mean(conv_list)),
                       'mean_rounds':    float(np.mean(rounds_list)),
                       'max_rounds':     int(np.max(rounds_list))}
                append_row(OUTPUT_FILE, row)
                done += 1
                elapsed = time.time() - t0
                print(f"[{done:5d}/{total}] n={n:2d} tech={tech_idx:2d} same_tech_dif_init  "
                      f"diversity={diversity:.3f}  conv={row['frac_converged']:.2f}  "
                      f"rounds={row['mean_rounds']:.1f}/{row['max_rounds']}  ({elapsed:.1f}s)")

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
    parser.add_argument('--base_seed', type=int, default=None,
                        help='Monte Carlo offset; different values give independent RNG streams')
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
    if args.base_seed is not None:
        BASE_SEED = args.base_seed
    if args.output is not None:
        OUTPUT_FILE = args.output

    run_study()
