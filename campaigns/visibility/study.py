"""
Visibility study: at fixed economic parameters, sweep tier visibility tau and
record convergence outcomes per (tech_matrix, tau) cell.

Loop structure (different from diversity_study.py):
    for tech_idx in 0..TECH_PER_JOB:
        build economy + initial network (deterministic from tech_seed)
        for tau in TAU_VALUES:
            build tier array (homogeneous OR lognormal heterogeneous)
            run one simulation
            log one CSV row

This is one independent simulation per (tech_idx, tau), with the same
underlying economy (a, b, z, AiSi, Wbar) and the same initial network reused
across tau values. That gives paired statistics across tau within each tech
matrix.

Resume key: (n, tech_idx, tier_mean, tier_std). Re-running with the same
output file skips already-done cells.

CSV columns: see CSV_FIELDNAMES below. There is no `series` column and no
`diversity` column (compared to diversity_study.py).
"""
import argparse
import csv
import json
import os
import random
import sys
import time

import numpy as np

# Allow `python scripts/visibility_study.py` from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

from rewiring.networks import generate_base_network, generate_random_initial_network
from rewiring.parameters import generate_a_parameter, generate_parameter
from rewiring.simulation import run_unified_simulation


# =============================================================================
# DEFAULT CONFIGURATION (overridden by CLI args)
# =============================================================================

N = 100
C = 4
CC = 4
AISI_SPREAD = 0.05
SIGMA_W = 0.05
MAX_SWAPS = 1
NB_ROUNDS = 200

A_CONFIG = {'mode': 'uniform', 'min': 0.4, 'max': 0.6}
B_CONFIG = {'mode': 'uniform', 'min': 0.9, 'max': 1.1}
Z_CONFIG = {'mode': 'uniform', 'min': 0.9, 'max': 1.1}

# tau axis to sweep on each tech matrix.
TAU_VALUES = [0, 1, 2, 3, 4, 5, 6]

# Heterogeneity: 'homo'   -> tier_arr = full(n, round(tau)).
#                'hetero' -> tier_arr drawn from a per-firm distribution
#                            controlled by TIER_DIST below.
TAU_MODE = 'homo'

# Hetero-tau distribution: 'poisson' -> Poisson(lambda = tau_mean) (new default).
#                          'lognormal' -> legacy lognormal(mean=tau_mean,
#                                                          std=tau_std).
TIER_DIST = 'poisson'

TECH_PER_JOB = 2
BASE_SEED = 0          # Monte Carlo offset; unique tech_seed = BASE_SEED * 10000 + tech_idx.

OUTPUT_FILE = "visibility_results.csv"


# =============================================================================
# TIER ARRAY HELPER
# =============================================================================

def build_tier_array(n, tier_mean, tier_std, rng, tier_dist='poisson'):
    """Return an int array of length n: tier visibility for each firm.

    Homogeneous branch (tier_std <= 0): constant array = round(tier_mean),
    regardless of tier_dist.

    Heterogeneous branch (tier_std > 0):
      tier_dist='poisson'  : tier_i ~ Poisson(lambda=tier_mean)  (new default)
      tier_dist='lognormal': tier_i ~ round(LogNormal) with target
                              mean = tier_mean, std = tier_std  (legacy)

    For Poisson, tier_std is ignored (the std is automatically sqrt(lambda)).
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
# CSV
# =============================================================================

CSV_FIELDNAMES = [
    'n', 'tech_idx', 'tier_mean', 'tier_std', 'tau_mode', 'tier_dist',
    # One sim per row, so frac_converged in {0, 1} (and frac_cycled in {0, 1}).
    # cycle_period: None = truncated; 1 = strict converge; 2..MAX_CYCLE_PERIOD = cycle.
    'converged', 'cycle_period', 'rounds', 'total_rewirings', 'swaps_per_firm',
    'final_utility', 'initial_utility',
    'U_T',  # aggregate household log-utility at termination, -sum_i log(P_i)
    # Fixed parameters of the cell
    'c', 'cc', 'aisi_spread', 'sigma_w', 'max_swaps', 'nb_rounds',
    'a_config', 'b_config', 'z_config',
    'mode',
    'base_seed', 'tech_seed', 'perm_seed',
]


def load_existing_keys(filepath):
    """Return set of (n, tech_idx, tier_mean, tier_std, tau_mode, tier_dist)
    rows already in CSV. The resume key now includes tier_dist so re-runs
    with a different distribution don't collide with old rows.
    """
    keys = set()
    if not os.path.exists(filepath):
        return keys
    with open(filepath, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            keys.add((
                int(row['n']),
                int(row['tech_idx']),
                float(row['tier_mean']),
                float(row['tier_std']),
                row['tau_mode'],
                row.get('tier_dist', 'lognormal'),  # default for legacy CSVs
            ))
    return keys


def append_row(filepath, row_dict):
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
    existing = load_existing_keys(OUTPUT_FILE)
    total_cells = TECH_PER_JOB * len(TAU_VALUES)
    done = 0

    print(f"Visibility study  n={N}  tech_per_job={TECH_PER_JOB}  tau={TAU_VALUES}  tau_mode={TAU_MODE}  tier_dist={TIER_DIST}")
    print(f"  Network : c={C}, cc={CC}, aisi_spread={AISI_SPREAD}, sigma_w={SIGMA_W}")
    print(f"  Economic: a={A_CONFIG}, b={B_CONFIG}, z={Z_CONFIG}")
    print(f"  Sim     : nb_rounds={NB_ROUNDS}, max_swaps={MAX_SWAPS}, mode=limited")
    print(f"  Output  : {OUTPUT_FILE}")
    print("=" * 70)

    for tech_idx in range(TECH_PER_JOB):
        tech_seed = BASE_SEED * 10000 + tech_idx
        seed_off = BASE_SEED * 10000

        # Economy parameters (deterministic from tech_seed)
        random.seed(tech_seed)
        np.random.seed(tech_seed)
        b = generate_parameter(B_CONFIG, N, param_name='b', verbose=False)
        a = generate_a_parameter(A_CONFIG, b, N, verbose=False)
        z = generate_parameter(Z_CONFIG, N, param_name='z', verbose=False)

        # Tech matrix (deterministic from tech_seed)
        base_state = generate_base_network(
            N, C, CC, AISI_SPREAD, seed=tech_seed,
            a=a, b=b, sigma_w=SIGMA_W,
        )
        Wbar = base_state['Wbar']
        AiSi = base_state['AiSi']

        # Initial network: drawn ONCE per tech_idx from a deterministic offset
        # seed and REUSED across all tau values. This gives paired "same start,
        # different tau" comparison within each tech matrix.
        init_state = generate_random_initial_network(
            N, Wbar, AiSi, seed=seed_off + 1_000_000 + tech_idx,
        )

        # Tier RNG, also reused across tau values within this tech matrix.
        # Different tau values pull fresh draws from this generator, so the
        # heterogeneous tier_arr varies by tau even with the same RNG.
        tier_rng = np.random.default_rng(tech_seed + 7919)

        common = {
            'n': N, 'tech_idx': tech_idx,
            'c': C, 'cc': CC, 'aisi_spread': AISI_SPREAD, 'sigma_w': SIGMA_W,
            'max_swaps': MAX_SWAPS, 'nb_rounds': NB_ROUNDS,
            'a_config': json.dumps(A_CONFIG),
            'b_config': json.dumps(B_CONFIG),
            'z_config': json.dumps(Z_CONFIG),
            'mode': 'limited',
            'tau_mode': TAU_MODE,
            'tier_dist': TIER_DIST,
            'base_seed': BASE_SEED,
            'tech_seed': tech_seed,
        }

        for tau_idx, tau in enumerate(TAU_VALUES):
            tier_mean = float(tau)
            tier_std = float(tau) if TAU_MODE == 'hetero' else 0.0
            key = (N, tech_idx, tier_mean, tier_std, TAU_MODE, TIER_DIST)
            if key in existing:
                done += 1
                continue

            tier_arr = build_tier_array(N, tier_mean, tier_std, tier_rng,
                                        tier_dist=TIER_DIST)
            perm_seed = tech_seed * 100 + tau_idx

            t0 = time.time()
            result = run_unified_simulation(
                init_state, a, b, z, mode='limited',
                seed=perm_seed,
                max_swaps=MAX_SWAPS,
                nb_rounds=NB_ROUNDS,
                tier=tier_arr,
            )
            elapsed = time.time() - t0

            cycle_period = result.get('cycle_period')
            P_T = np.asarray(result['final_prices'], dtype=float)
            P_T_pos = P_T[P_T > 0]
            U_T = float(-np.sum(np.log(P_T_pos))) if P_T_pos.size else float('nan')
            row = {
                **common,
                'tier_mean': tier_mean,
                'tier_std': tier_std,
                'converged':       int(bool(result['converged'])),
                'cycle_period':    (int(cycle_period) if cycle_period is not None else ''),
                'rounds':          int(result['rounds']),
                'total_rewirings': int(result.get('total_rewirings', 0)),
                'swaps_per_firm':  float(int(result.get('total_rewirings', 0))) / float(N),
                'final_utility':   float(result['final_utility']),
                'initial_utility': float(result['initial_utility']),
                'U_T':             U_T,
                'perm_seed':       perm_seed,
            }
            append_row(OUTPUT_FILE, row)
            done += 1

            cp_str = f"k={cycle_period}" if cycle_period is not None and cycle_period >= 2 else (
                "conv" if cycle_period == 1 else "trunc"
            )
            print(f"[{done:4d}/{total_cells}] tech={tech_idx:2d} tau={tau} "
                  f"-> {cp_str:6s}  rounds={row['rounds']:3d}  "
                  f"swaps/firm={row['swaps_per_firm']:.2f}  ({elapsed:.1f}s)")

    print("=" * 70)
    print(f"Done. Results saved to {OUTPUT_FILE}")


# =============================================================================
# CLI
# =============================================================================

def parse_config_arg(s):
    parts = s.split(':')
    mode = parts[0]
    if mode == 'homogeneous':
        return {'mode': 'homogeneous', 'value': float(parts[1])}
    elif mode == 'uniform':
        return {'mode': 'uniform', 'min': float(parts[1]), 'max': float(parts[2])}
    raise ValueError(f"Unknown config mode: {mode}")


def parse_tau_values(s):
    return [int(x) for x in s.split(',') if x.strip()]


def parse_args():
    parser = argparse.ArgumentParser(description='Visibility study')
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--tech_per_job', type=int, default=None)
    parser.add_argument('--nb_rounds', type=int, default=None)
    parser.add_argument('--cc', type=int, default=None)
    parser.add_argument('--aisi_spread', type=float, default=None)
    parser.add_argument('--sigma_w', type=float, default=None)
    parser.add_argument('--max_swaps', type=int, default=None)
    parser.add_argument('--a_config', type=str, default=None)
    parser.add_argument('--b_config', type=str, default=None)
    parser.add_argument('--z_config', type=str, default=None)
    parser.add_argument('--tau_values', type=str, default=None,
                        help='Comma-separated, e.g. "0,1,2,3,4,5,6"')
    parser.add_argument('--tau_mode', type=str, default=None,
                        choices=['homo', 'hetero'],
                        help='homo: tier_arr = full(n, tau).  '
                             'hetero: per-firm draw from --tier_dist.')
    parser.add_argument('--tier_dist', type=str, default=None,
                        choices=['poisson', 'lognormal'],
                        help='Distribution used in hetero mode (default: '
                             'poisson; legacy: lognormal).')
    parser.add_argument('--base_seed', type=int, default=None)
    parser.add_argument('--output', type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.n is not None: N = args.n
    if args.tech_per_job is not None: TECH_PER_JOB = args.tech_per_job
    if args.nb_rounds is not None: NB_ROUNDS = args.nb_rounds
    if args.cc is not None: CC = args.cc
    if args.aisi_spread is not None: AISI_SPREAD = args.aisi_spread
    if args.sigma_w is not None: SIGMA_W = args.sigma_w
    if args.max_swaps is not None: MAX_SWAPS = args.max_swaps
    if args.a_config is not None: A_CONFIG = parse_config_arg(args.a_config)
    if args.b_config is not None: B_CONFIG = parse_config_arg(args.b_config)
    if args.z_config is not None: Z_CONFIG = parse_config_arg(args.z_config)
    if args.tau_values is not None: TAU_VALUES = parse_tau_values(args.tau_values)
    if args.tau_mode is not None: TAU_MODE = args.tau_mode
    if args.tier_dist is not None: TIER_DIST = args.tier_dist
    if args.base_seed is not None: BASE_SEED = args.base_seed
    if args.output is not None: OUTPUT_FILE = args.output

    run_study()
