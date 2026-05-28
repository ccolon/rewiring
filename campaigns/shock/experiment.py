"""Shock experiment: shutdown / reintroduction of a single firm in `mode='full'`.

Runs three sequential phases on a fixed economy, using the same
`run_unified_simulation` driver:

    Phase 1:  random initial network -> M1 (first stable configuration)
    Phase 2:  remove a randomly chosen firm i*; force-replace it in every
              current supplier list; rerun with i* forbidden as a supplier
              -> M2 (second stable configuration after the shock)
    Phase 3:  re-introduce i*; rerun from M2
              -> M3 (third stable configuration after re-introduction)

The "removal" is implemented by:
  (a) force-replacing i* in every firm's current supplier set with a random
      element of that firm's allowed pool (alternates), and
  (b) restricting each firm's alternates list to exclude i* during phase 2.
This matches the manuscript's prose ("clients of the disappeared firm need
new suppliers from their set of possible suppliers, S_i_bar") without the
numerical fragility of literally setting z_i = 0.

Output: a pickle with the full per-phase trace, distance-from-baseline
time series, and metadata. The companion `plot_shock.py` renders figures.

Usage:
    python scripts/shock_experiment.py                       # default 50/4/4 b=0.9 hom
    python scripts/shock_experiment.py --seed 7              # different economy
    python scripts/shock_experiment.py --aisi 0.05 --b_config uniform:0.9:1.1
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


# -----------------------------------------------------------------------------
# Parameter parsing helpers (mirrors diversity_study.py)
# -----------------------------------------------------------------------------

def _parse_param(s, n, rng):
    """Parse 'homogeneous:V' or 'uniform:LO:HI' into a length-n array."""
    parts = s.split(':')
    mode = parts[0]
    if mode == 'homogeneous':
        return np.full(n, float(parts[1]))
    if mode == 'uniform':
        lo, hi = float(parts[1]), float(parts[2])
        return rng.uniform(lo, hi, n)
    raise ValueError(f"Unknown param mode: {mode!r}")


# -----------------------------------------------------------------------------
# Network-state manipulation
# -----------------------------------------------------------------------------

def _allowed_pool(network_state):
    """Per-firm set of all firms it is allowed to consider as suppliers
    (current + alternates)."""
    return [
        set(network_state['supplier_id_list'][i])
        | set(network_state['alternate_supplier_id_list'][i])
        for i in range(len(network_state['supplier_id_list']))
    ]


def _force_replace_supplier(suppliers, allowed_pool, victim, rng):
    """For every firm whose supplier set contains `victim`, swap it out for
    a random element of (allowed_pool - current - {victim, self}).

    Returns a new supplier list (does not mutate the input).
    """
    new = [list(s) for s in suppliers]
    n_replaced = 0
    for i, cur in enumerate(new):
        if victim in cur:
            cands = list(allowed_pool[i] - set(cur) - {victim, i})
            if cands:
                replacement = int(rng.choice(cands))
                cur.remove(victim)
                cur.append(replacement)
                cur.sort()
                n_replaced += 1
    return new, n_replaced


def _make_phase_state(base_ns, suppliers, allowed_pool, forbidden=None):
    """Return a network_state dict suitable for run_unified_simulation.

    Starts from `suppliers` (per-firm supplier list). Alternates are computed
    as (allowed_pool[i] - current - {forbidden, self}). Wbar / AiSi / nb_suppliers
    inherit from base_ns.
    """
    forbidden_set = set() if forbidden is None else {int(forbidden)}
    n = len(suppliers)
    alts = []
    for i in range(n):
        a = allowed_pool[i] - set(suppliers[i]) - forbidden_set - {i}
        alts.append(sorted(int(x) for x in a))

    M = np.zeros((n, n), dtype=np.float64)
    for buyer, ss in enumerate(suppliers):
        for s in ss:
            M[s, buyer] = 1.0
    return {
        'M0': M,
        'W0': M * base_ns['Wbar'],
        'Wbar': base_ns['Wbar'],
        'supplier_id_list': [list(s) for s in suppliers],
        'alternate_supplier_id_list': alts,
        'AiSi': base_ns['AiSi'],
        'nb_suppliers': np.array([len(s) for s in suppliers]),
    }


def _pick_firm_with_clients(suppliers, rng):
    """Pick a uniformly-random firm that appears in at least one supplier list."""
    n = len(suppliers)
    has_clients = np.zeros(n, dtype=bool)
    for ss in suppliers:
        for s in ss:
            has_clients[s] = True
    candidates = np.where(has_clients)[0]
    if len(candidates) == 0:
        raise RuntimeError("No firm has any client; cannot run shock experiment.")
    return int(rng.choice(candidates))


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------

def run_shock_experiment(seed, n=50, c=4, cc=4, max_swaps=1, mode='full',
                         a_config='homogeneous:0.5',
                         b_config='homogeneous:0.9',
                         z_config='homogeneous:1.0',
                         aisi_spread=0.0, sigma_w=0.0,
                         nb_rounds=200):
    rng = np.random.default_rng(seed)

    # Per-firm a / b / z arrays
    a = _parse_param(a_config, n, rng)
    b = _parse_param(b_config, n, rng)
    z = _parse_param(z_config, n, rng)

    # Build the base economy. seed=seed inside generate_base_network controls
    # SF-FA topology, AiSi multipliers, sigma_w noise, etc.
    base_ns = generate_base_network(
        n=n, c=c, cc=cc, aisi_spread=aisi_spread,
        seed=seed, a=a, b=b, sigma_w=sigma_w,
    )
    pool = _allowed_pool(base_ns)

    # Random initial supplier configuration on the same pool
    init_ns = generate_random_initial_network(
        n=n, Wbar=base_ns['Wbar'], AiSi=base_ns['AiSi'], seed=seed + 1000,
    )

    print(f"Phase 1: random init -> M1  (mode={mode}, max_swaps={max_swaps})")
    r1 = run_unified_simulation(
        init_ns, a, b, z, mode=mode, seed=seed + 2000,
        max_swaps=max_swaps, nb_rounds=nb_rounds, trace=True,
    )
    M1 = r1['final_supplier_list']
    print(f"  rounds={r1['rounds']}  total_rewirings={r1['total_rewirings']}  "
          f"converged={r1['converged']}")

    # Pick the firm to shutdown
    firm_idx = _pick_firm_with_clients(M1, rng)
    n_clients_M1 = sum(1 for ss in M1 if firm_idx in ss)
    print(f"\nShutdown firm {firm_idx} (currently supplying {n_clients_M1} clients)")

    # Force-replace firm_idx in M1
    M1_post_shock, n_replaced = _force_replace_supplier(M1, pool, firm_idx, rng)
    print(f"  forced-replaced firm {firm_idx} in {n_replaced} supplier list(s)")

    # Phase 2 state: starts at M1_post_shock, firm_idx is forbidden
    ns_phase2 = _make_phase_state(base_ns, M1_post_shock, pool, forbidden=firm_idx)

    print(f"\nPhase 2: M1' -> M2 (firm {firm_idx} forbidden as supplier)")
    r2 = run_unified_simulation(
        ns_phase2, a, b, z, mode=mode, seed=seed + 3000,
        max_swaps=max_swaps, nb_rounds=nb_rounds, trace=True,
    )
    M2 = r2['final_supplier_list']
    print(f"  rounds={r2['rounds']}  total_rewirings={r2['total_rewirings']}  "
          f"converged={r2['converged']}")

    # Phase 3 state: starts at M2, firm_idx allowed again
    ns_phase3 = _make_phase_state(base_ns, M2, pool, forbidden=None)

    print(f"\nPhase 3: M2 -> M3 (firm {firm_idx} re-introduced)")
    r3 = run_unified_simulation(
        ns_phase3, a, b, z, mode=mode, seed=seed + 4000,
        max_swaps=max_swaps, nb_rounds=nb_rounds, trace=True,
    )
    M3 = r3['final_supplier_list']
    print(f"  rounds={r3['rounds']}  total_rewirings={r3['total_rewirings']}  "
          f"converged={r3['converged']}")

    return {
        'config': dict(seed=seed, n=n, c=c, cc=cc, max_swaps=max_swaps, mode=mode,
                       a_config=a_config, b_config=b_config, z_config=z_config,
                       aisi_spread=aisi_spread, sigma_w=sigma_w, nb_rounds=nb_rounds),
        'a': a, 'b': b, 'z': z,
        'init_supplier_list': [list(s) for s in init_ns['supplier_id_list']],
        'firm_shutdown': firm_idx,
        'n_clients_at_shutdown': n_clients_M1,
        'M1_post_shock': M1_post_shock,
        'phase1': r1,
        'phase2': r2,
        'phase3': r3,
        'base_ns': {k: base_ns[k] for k in ('M0', 'Wbar', 'supplier_id_list',
                                             'alternate_supplier_id_list',
                                             'nb_suppliers')},
    }


def main():
    p = argparse.ArgumentParser(description='Single-firm shock / reintroduction experiment')
    p.add_argument('--seed', type=int, default=0)
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
    p.add_argument('--output', default=None,
                   help='output pickle path; default: results/shock/shock_seed{seed}.pkl')
    args = p.parse_args()

    if args.output is None:
        out_dir = os.path.join(REPO_ROOT, 'results', 'shock')
        os.makedirs(out_dir, exist_ok=True)
        args.output = os.path.join(out_dir, f'shock_seed{args.seed}.pkl')

    result = run_shock_experiment(
        seed=args.seed, n=args.n, c=args.c, cc=args.cc,
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
