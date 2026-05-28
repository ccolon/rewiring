"""Repeated-shock experiment: K cycles of removal + reintroduction.

The appendix-figure analogue of `shock_experiment.py`. Each cycle picks a
fresh random firm, runs a shutdown phase to a new stable configuration, then
a reintroduction phase to another stable configuration. Across many cycles
we see cumulative drift through the space of stable configurations.

Output: pickle with the per-phase trace, the cycle records (firm_idx,
M_shut, M_full per cycle), and the initial / first-stable configurations.

Usage:
    python scripts/shock_loop_experiment.py                           # K=10, default op-point
    python scripts/shock_loop_experiment.py --n_shocks 5 --seed 7
    python scripts/shock_loop_experiment.py --aisi 0.05 --b_config uniform:0.9:1.1
"""
import argparse
import os
import pickle
import sys

import numpy as np

# This script lives at campaigns/shock/, so REPO_ROOT is 3 levels up.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.networks import generate_base_network, generate_random_initial_network
from rewiring.simulation import run_unified_simulation
from campaigns.shock.experiment import (
    _allowed_pool,
    _force_replace_supplier,
    _make_phase_state,
    _parse_param,
    _pick_firm_with_clients,
)


def run_shock_loop_experiment(seed, n_shocks=10, n=50, c=4, cc=4, max_swaps=1,
                              mode='full',
                              a_config='homogeneous:0.5',
                              b_config='homogeneous:0.9',
                              z_config='homogeneous:1.0',
                              aisi_spread=0.0, sigma_w=0.0,
                              nb_rounds=200):
    rng = np.random.default_rng(seed)

    a = _parse_param(a_config, n, rng)
    b = _parse_param(b_config, n, rng)
    z = _parse_param(z_config, n, rng)

    base_ns = generate_base_network(
        n=n, c=c, cc=cc, aisi_spread=aisi_spread,
        seed=seed, a=a, b=b, sigma_w=sigma_w,
    )
    pool = _allowed_pool(base_ns)

    init_ns = generate_random_initial_network(
        n=n, Wbar=base_ns['Wbar'], AiSi=base_ns['AiSi'], seed=seed + 1000,
    )

    print(f"Phase 0: random init -> M_init  (mode={mode}, max_swaps={max_swaps})")
    r0 = run_unified_simulation(
        init_ns, a, b, z, mode=mode, seed=seed + 2000,
        max_swaps=max_swaps, nb_rounds=nb_rounds, trace=True,
    )
    M_init_stable = r0['final_supplier_list']
    print(f"  rounds={r0['rounds']}  total_rewirings={r0['total_rewirings']}  "
          f"converged={r0['converged']}")

    cycles = []
    current_suppliers = M_init_stable
    for k in range(1, n_shocks + 1):
        # Pick firm to shutdown (from those with at least one client in the
        # current configuration).
        firm_idx = _pick_firm_with_clients(current_suppliers, rng)
        n_clients = sum(1 for ss in current_suppliers if firm_idx in ss)

        # Shutdown phase: force-replace, restrict alternates, rerun
        post_shock_suppliers, n_replaced = _force_replace_supplier(
            current_suppliers, pool, firm_idx, rng,
        )
        ns_shut = _make_phase_state(base_ns, post_shock_suppliers, pool,
                                    forbidden=firm_idx)
        r_shut = run_unified_simulation(
            ns_shut, a, b, z, mode=mode, seed=seed + 3000 + k * 100,
            max_swaps=max_swaps, nb_rounds=nb_rounds, trace=True,
        )
        M_shut = r_shut['final_supplier_list']

        # Reintroduction phase: starts at M_shut, firm_idx allowed again
        ns_full = _make_phase_state(base_ns, M_shut, pool, forbidden=None)
        r_full = run_unified_simulation(
            ns_full, a, b, z, mode=mode, seed=seed + 4000 + k * 100,
            max_swaps=max_swaps, nb_rounds=nb_rounds, trace=True,
        )
        M_full = r_full['final_supplier_list']

        print(f"  cycle {k:>2d}: shutdown firm {firm_idx:>3d} ({n_clients} clients) "
              f"-> rewirings shut={r_shut['total_rewirings']:>3d}, "
              f"reintro={r_full['total_rewirings']:>3d}")

        cycles.append({
            'k': k,
            'firm_idx': firm_idx,
            'n_clients_at_shutdown': n_clients,
            'n_replaced': n_replaced,
            'shutdown': r_shut,
            'reintroduction': r_full,
            'M_shut': M_shut,
            'M_full': M_full,
        })
        current_suppliers = M_full

    return {
        'config': dict(seed=seed, n_shocks=n_shocks, n=n, c=c, cc=cc,
                       max_swaps=max_swaps, mode=mode,
                       a_config=a_config, b_config=b_config, z_config=z_config,
                       aisi_spread=aisi_spread, sigma_w=sigma_w,
                       nb_rounds=nb_rounds),
        'a': a, 'b': b, 'z': z,
        'init_supplier_list': [list(s) for s in init_ns['supplier_id_list']],
        'phase_init': r0,
        'M_init_stable': M_init_stable,
        'cycles': cycles,
    }


def main():
    p = argparse.ArgumentParser(description='K-cycle removal/reintroduction shock experiment')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n_shocks', type=int, default=10)
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--c', type=int, default=4)
    p.add_argument('--cc', type=int, default=4)
    p.add_argument('--max_swaps', type=int, default=1)
    p.add_argument('--mode', choices=['full', 'aa', 'limited', 'naive_limited'],
                   default='full')
    p.add_argument('--a_config', default='homogeneous:0.5')
    p.add_argument('--b_config', default='homogeneous:0.9')
    p.add_argument('--z_config', default='homogeneous:1.0')
    p.add_argument('--aisi', type=float, default=0.0)
    p.add_argument('--sigma_w', type=float, default=0.0)
    p.add_argument('--nb_rounds', type=int, default=200)
    p.add_argument('--output', default=None)
    args = p.parse_args()

    if args.output is None:
        out_dir = os.path.join(REPO_ROOT, 'results', 'shock')
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir,
                                   f'shock_loop_K{args.n_shocks}_seed{args.seed}.pkl')

    result = run_shock_loop_experiment(
        seed=args.seed, n_shocks=args.n_shocks,
        n=args.n, c=args.c, cc=args.cc,
        max_swaps=args.max_swaps, mode=args.mode,
        a_config=args.a_config, b_config=args.b_config, z_config=args.z_config,
        aisi_spread=args.aisi, sigma_w=args.sigma_w,
        nb_rounds=args.nb_rounds,
    )

    with open(args.output, 'wb') as f:
        pickle.dump(result, f)
    print(f"\nWrote {args.output}")


if __name__ == '__main__':
    main()
