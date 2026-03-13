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

import csv
import json
import os
import random
import time

import numpy as np

from test_convergence import (
    generate_base_network,
    generate_random_initial_network,
    run_full_simulation,
)
from utils import generate_parameter, generate_a_parameter


# =============================================================================
# CONFIGURATION  (edit here before each run)
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
    'c', 'cc', 'aisi_spread', 'sigma_w', 'max_swaps', 'nb_rounds', 'n_trials',
    'a_config', 'b_config', 'z_config',
    'tech_seed',
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
            # tech_seed = tech_idx so seeds 0..N_TECH_MATRICES-1 are used
            # independently for each n.
            # ------------------------------------------------------------------
            tech_seed = tech_idx

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
                'tech_seed': tech_seed,
            }

            # ------------------------------------------------------------------
            # same_tech_same_init — same initial network, different permutation orders
            #   perm_seed = 1000 + trial  controls np.random.permutation inside
            #   the simulation (evaluation order of firms each round).
            # ------------------------------------------------------------------
            if blue_key not in existing_keys:
                t0 = time.time()
                final_lists = []
                for trial in range(N_TRIALS):
                    result = run_full_simulation(
                        base_state, a, b, z,
                        seed=1000 + trial,
                        max_swaps=MAX_SWAPS,
                        nb_rounds=NB_ROUNDS,
                    )
                    final_lists.append(result['final_supplier_list'])

                diversity = compute_diversity(final_lists)
                append_row(OUTPUT_FILE, {**common, 'series': 'same_tech_same_init', 'diversity': diversity})
                done += 1
                elapsed = time.time() - t0
                print(f"[{done:5d}/{total}] n={n:2d} tech={tech_idx:2d} same_tech_same_init  "
                      f"diversity={diversity:.3f}  ({elapsed:.1f}s)")

            # ------------------------------------------------------------------
            # same_tech_dif_init — different initial networks, different permutation orders
            #   init_seed = 2000 + trial  controls the random initial supplier
            #   selection via generate_random_initial_network.
            #   perm_seed = 3000 + trial  controls firm evaluation order.
            # ------------------------------------------------------------------
            if red_key not in existing_keys:
                t0 = time.time()
                final_lists = []
                for trial in range(N_TRIALS):
                    trial_state = generate_random_initial_network(
                        n, Wbar, AiSi, seed=2000 + trial,
                    )
                    result = run_full_simulation(
                        trial_state, a, b, z,
                        seed=3000 + trial,
                        max_swaps=MAX_SWAPS,
                        nb_rounds=NB_ROUNDS,
                    )
                    final_lists.append(result['final_supplier_list'])

                diversity = compute_diversity(final_lists)
                append_row(OUTPUT_FILE, {**common, 'series': 'same_tech_dif_init', 'diversity': diversity})
                done += 1
                elapsed = time.time() - t0
                print(f"[{done:5d}/{total}] n={n:2d} tech={tech_idx:2d} same_tech_dif_init  "
                      f"diversity={diversity:.3f}  ({elapsed:.1f}s)")

    print("=" * 70)
    print(f"Done. Results saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_study()
