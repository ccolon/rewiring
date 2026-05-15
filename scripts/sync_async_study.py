"""Sync vs async rewiring campaign (manuscript Campaign 2, app:sync_async).

For each of 50 distinct (W_bar, S^(0)) pairs, run:
  - 1 synchronous (Jacobi) trajectory.  Deterministic conditional on the pair.
  - 50 asynchronous (Gauss-Seidel) trajectories with different
    firm-activation orders (perm_seed).

For every (p, k) we record the cosine network distance between the async
fixed point and the deterministic synchronous fixed point of the same pair.

Parameter point: same as P2 of the welfare-dispersion campaign --
    CRS (b_i = 1), kappa = 1, Delta_A = 0.05, sigma_w = Delta_z = 0,
    full-GE anticipation, n = 50, c = c' = 4, a_i = 0.5, z_i = 1.

Output CSV columns (per (p, k) async run; one extra row per pair for the
sync result with async_idx = -1):
    pair_idx, async_idx, tech_seed, init_seed, perm_seed,
    distance, converged, rounds, run_kind

Distance to the sync fixed point is computed in-script using the
manuscript's cosine network distance (eq:network_distance):
    d(M_s, M_t) = 1 - <M_s, M_t> / sqrt(<M_s,M_s>*<M_t,M_t>)
For active-supplier edge sets E_s, E_t with the same edge cardinality (the
case in this model), this reduces to 1 - |E_s & E_t| / |E_s|.

Usage:
    python scripts/sync_async_study.py --n_pairs 50 --n_async 50 \
        --base_seed 1800 --output results/sync_async/sync_async_seed1800.csv
"""
import argparse
import csv
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.networks import (
    generate_base_network,
    generate_random_initial_network,
)
from rewiring.simulation import run_unified_simulation


# Parameter point (same as welfare-dispersion P2).
N = 50
C = 4
CC = 4
A_VALUE = 0.5
B_VALUE = 1.0   # CRS
Z_VALUE = 1.0
AISI = 0.05
SIGMA_W = 0.0
KAPPA = 1
NB_ROUNDS = 200


CSV_FIELDS = ['pair_idx', 'async_idx', 'tech_seed', 'init_seed', 'perm_seed',
              'distance', 'converged', 'rounds', 'run_kind']


def supplier_list_to_edge_set(supplier_list):
    """Convert [[s, ...], ...] -> frozenset of (supplier, buyer) tuples."""
    out = set()
    for buyer, suppliers in enumerate(supplier_list):
        for s in suppliers:
            out.add((int(s), int(buyer)))
    return frozenset(out)


def cosine_edge_distance(e1, e2):
    """1 - Ochiai (cosine) distance between two binary edge sets."""
    if not e1 or not e2:
        return 1.0
    return 1.0 - len(e1 & e2) / math.sqrt(len(e1) * len(e2))


def existing_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            keys.add((int(row['pair_idx']), int(row['async_idx'])))
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n_pairs', type=int, default=50)
    p.add_argument('--n_async', type=int, default=50)
    p.add_argument('--base_seed', type=int, default=1800)
    p.add_argument('--nb_rounds', type=int, default=NB_ROUNDS)
    p.add_argument('--output', required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    SEED_MOD = 2**31 - 1
    print(f"Sync/async study | n_pairs={args.n_pairs}  n_async={args.n_async}")
    print(f"  Point: n={N}, c={C}, cc={CC}, kappa={KAPPA}, aisi={AISI}, "
          f"sigma_w={SIGMA_W}, b=hom {B_VALUE}, a=hom {A_VALUE}, z=hom {Z_VALUE}")
    print(f"  Output: {args.output}")

    done = existing_keys(args.output)
    write_header = not os.path.exists(args.output)
    f_out = open(args.output, 'a', newline='')
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    a_arr = np.full(N, A_VALUE)
    b_arr = np.full(N, B_VALUE)
    z_arr = np.full(N, Z_VALUE)

    n_done = 0
    for pair_idx in range(args.n_pairs):
        tech_seed = (args.base_seed * 10_000 + pair_idx) % SEED_MOD
        init_seed = (tech_seed * 100 + 1) % SEED_MOD  # one init per pair

        base_ns = generate_base_network(
            n=N, c=C, cc=CC, aisi_spread=AISI,
            seed=int(tech_seed), a=a_arr, b=b_arr, sigma_w=SIGMA_W,
        )
        init_ns = generate_random_initial_network(
            n=N, Wbar=base_ns['Wbar'], AiSi=base_ns['AiSi'],
            seed=int(init_seed),
        )

        # ---- Synchronous trajectory (deterministic conditional on the pair).
        sync_key = (pair_idx, -1)
        sync_result = None
        if sync_key not in done:
            sync_result = run_unified_simulation(
                init_ns, a_arr, b_arr, z_arr,
                mode='full', max_swaps=KAPPA, nb_rounds=args.nb_rounds,
                synchronous=True,
            )
            sync_edges = supplier_list_to_edge_set(
                sync_result['final_supplier_list'])
            writer.writerow({
                'pair_idx':  pair_idx,
                'async_idx': -1,
                'tech_seed': int(tech_seed),
                'init_seed': int(init_seed),
                'perm_seed': '',
                'distance':  0.0,
                'converged': int(bool(sync_result['converged'])),
                'rounds':    int(sync_result['rounds']),
                'run_kind':  'sync',
            })
            n_done += 1
        else:
            # Recompute the sync fixed point for distance reference (it's
            # cheap and we already need it to compute async distances).
            sync_result = run_unified_simulation(
                init_ns, a_arr, b_arr, z_arr,
                mode='full', max_swaps=KAPPA, nb_rounds=args.nb_rounds,
                synchronous=True,
            )
            sync_edges = supplier_list_to_edge_set(
                sync_result['final_supplier_list'])

        # ---- 50 asynchronous trajectories.
        for k in range(args.n_async):
            if (pair_idx, k) in done:
                continue
            perm_seed = (init_seed + 1000 + k) % SEED_MOD
            async_result = run_unified_simulation(
                init_ns, a_arr, b_arr, z_arr,
                mode='full', seed=int(perm_seed),
                max_swaps=KAPPA, nb_rounds=args.nb_rounds,
                synchronous=False,
            )
            async_edges = supplier_list_to_edge_set(
                async_result['final_supplier_list'])
            d = cosine_edge_distance(async_edges, sync_edges)
            writer.writerow({
                'pair_idx':  pair_idx,
                'async_idx': k,
                'tech_seed': int(tech_seed),
                'init_seed': int(init_seed),
                'perm_seed': int(perm_seed),
                'distance':  float(d),
                'converged': int(bool(async_result['converged'])),
                'rounds':    int(async_result['rounds']),
                'run_kind':  'async',
            })
            n_done += 1
            if n_done % 25 == 0:
                f_out.flush()
                print(f"  ... {n_done} runs done (pair {pair_idx+1}/"
                      f"{args.n_pairs})")

    f_out.close()
    print(f"Done. {n_done} runs written to {args.output}")


if __name__ == '__main__':
    main()
