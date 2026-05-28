"""Welfare-dispersion campaign (manuscript §4.2.2, tab:welfare_dispersion).

For one of the five parameter points P0..P4, run 2500 trials = (tech matrices)
x (random initial networks) and record the trial-level aggregate household
log-utility U_T at the rewiring fixed point.

The seed scheme is shared across points: at a given --base_seed, the
tech_seed for tech_idx and init_seed for (tech_idx, init_idx) are identical
across point invocations. The analyzer then joins each point's CSV against
P0's CSV on (tech_seed, init_seed) to attach U_AA for the same
(W_bar, S^(0)) pair (P0 *is* the AA baseline: CRS, kappa = c', no dispersion).

Output: one CSV per (point, seed-batch).  Minimal schema, per design:
    tech_seed, init_seed, point, U_T, converged, R

Fallback: when a run hits R_max without converging or cycling, U_T is taken
as -sum(log(P_T)) at the LAST observed state (whatever the simulator left in
final_prices), per the spec's "use the last observed sum_p value as a fallback".

Usage:
    python scripts/welfare_dispersion_study.py --point P2 \
        --tech_per_job 10 --inits_per_tech 50 --base_seed 1700 \
        --output results/welfare_dispersion/welfare_P2_seed1700.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

# This script lives at campaigns/welfare_dispersion/, so REPO_ROOT is 3 levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.networks import (
    generate_base_network,
    generate_random_initial_network,
)
from rewiring.simulation import run_unified_simulation


# Parameter-point grid (§4.2.2 of the manuscript). All points share
# n=50, c=c'=4, a_i=0.5, z_i=1, full-GE anticipation, R_max=200.
POINTS = {
    'P0': {  # control: AA baseline, single fixed point
        'b_value': 1.0, 'kappa': 4, 'aisi': 0.0, 'sigma_w': 0.0, 'delta_z': 0.0,
        'b_hetero': False,
    },
    'P1': {  # pure search friction
        'b_value': 1.0, 'kappa': 1, 'aisi': 0.0, 'sigma_w': 0.0, 'delta_z': 0.0,
        'b_hetero': False,
    },
    'P2': {  # Delta_A traps
        'b_value': 1.0, 'kappa': 1, 'aisi': 0.05, 'sigma_w': 0.0, 'delta_z': 0.0,
        'b_hetero': False,
    },
    'P3': {  # non-CRS structural
        'b_value': None, 'kappa': 4, 'aisi': 0.0, 'sigma_w': 0.0, 'delta_z': 0.0,
        'b_hetero': True,  # b_i ~ U[0.9, 1.1]
    },
    'P4': {  # compounded
        'b_value': None, 'kappa': 1, 'aisi': 0.05, 'sigma_w': 0.05, 'delta_z': 0.0,
        'b_hetero': True,
    },
}

N = 50
C = 4
CC = 4
A_VALUE = 0.5
Z_VALUE = 1.0
NB_ROUNDS = 200


CSV_FIELDS = ['tech_seed', 'init_seed', 'point', 'U_T', 'converged', 'R']


def _draw_b(point_cfg, n, rng):
    if point_cfg['b_hetero']:
        return rng.uniform(0.9, 1.1, n)
    return np.full(n, float(point_cfg['b_value']))


def _draw_z(point_cfg, n, rng):
    dz = point_cfg['delta_z']
    if dz <= 0:
        return np.full(n, Z_VALUE)
    return rng.uniform(1 - dz, 1 + dz, n)


def _agg_log_utility(final_prices):
    """U_T = -sum_i log(P_i_T), defensive against non-positive prices."""
    P = np.asarray(final_prices, dtype=float)
    P = P[P > 0]
    if P.size == 0:
        return float('nan')
    return float(-np.sum(np.log(P)))


def existing_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            keys.add((int(row['tech_seed']), int(row['init_seed'])))
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--point', required=True, choices=sorted(POINTS.keys()),
                   help='Parameter point P0..P4.')
    p.add_argument('--tech_per_job', type=int, default=10)
    p.add_argument('--inits_per_tech', type=int, default=50)
    p.add_argument('--base_seed', type=int, default=1700,
                   help='Outer seed; tech_seed = base_seed*10000 + tech_idx, '
                        'init_seed = tech_seed*100 + init_idx.')
    p.add_argument('--nb_rounds', type=int, default=NB_ROUNDS)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    point_cfg = POINTS[args.point]
    SEED_MOD = 2**31 - 1

    print(f"Welfare-dispersion study | point={args.point} | "
          f"b_hetero={point_cfg['b_hetero']}  b_value={point_cfg['b_value']}  "
          f"kappa={point_cfg['kappa']}  aisi={point_cfg['aisi']}  "
          f"sw={point_cfg['sigma_w']}  delta_z={point_cfg['delta_z']}")
    print(f"  budget: {args.tech_per_job} tech x {args.inits_per_tech} inits = "
          f"{args.tech_per_job * args.inits_per_tech} trials")
    print(f"  output: {args.output}")

    done = existing_keys(args.output)
    write_header = not os.path.exists(args.output)
    f_out = open(args.output, 'a', newline='')
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    n_done = 0
    n_unconv = 0
    for tech_idx in range(args.tech_per_job):
        tech_seed = (args.base_seed * 10_000 + tech_idx) % SEED_MOD
        rng_tech = np.random.default_rng(tech_seed)
        a_arr = np.full(N, A_VALUE)
        b_arr = _draw_b(point_cfg, N, rng_tech)
        z_arr = _draw_z(point_cfg, N, rng_tech)

        base_ns = generate_base_network(
            n=N, c=C, cc=CC,
            aisi_spread=point_cfg['aisi'],
            seed=int(tech_seed),
            a=a_arr, b=b_arr,
            sigma_w=point_cfg['sigma_w'],
        )

        for init_idx in range(args.inits_per_tech):
            init_seed = (tech_seed * 100 + init_idx) % SEED_MOD
            if (int(tech_seed), int(init_seed)) in done:
                continue

            init_ns = generate_random_initial_network(
                n=N, Wbar=base_ns['Wbar'], AiSi=base_ns['AiSi'],
                seed=int(init_seed),
            )
            result = run_unified_simulation(
                init_ns, a_arr, b_arr, z_arr,
                mode='full',
                seed=int((init_seed + 31337) % SEED_MOD),
                max_swaps=point_cfg['kappa'],
                nb_rounds=args.nb_rounds,
            )
            # U_T uses final_prices regardless of convergence -- this gives the
            # "last observed" fallback when the run hits R_max.
            U_T = _agg_log_utility(result['final_prices'])
            converged = bool(result['converged'])
            if not converged:
                n_unconv += 1

            writer.writerow({
                'tech_seed': int(tech_seed),
                'init_seed': int(init_seed),
                'point':     args.point,
                'U_T':       U_T,
                'converged': int(converged),
                'R':         int(result['rounds']),
            })
            n_done += 1
            if n_done % 50 == 0:
                f_out.flush()
                print(f"  ... {n_done} trials done "
                      f"({n_unconv} non-converged so far)")

    f_out.close()
    print(f"Done. {n_done} trials written to {args.output}  "
          f"(non-converged: {n_unconv}).")


if __name__ == '__main__':
    main()
