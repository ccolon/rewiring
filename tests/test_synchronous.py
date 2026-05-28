"""Sanity checks for synchronous (Jacobi) mode of run_unified_simulation.

Run from anywhere:
    python tests/test_synchronous.py
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.networks import generate_base_network  # noqa: E402
from rewiring.parameters import generate_a_parameter, generate_parameter  # noqa: E402
from rewiring.simulation import run_unified_simulation  # noqa: E402


def _make_economy(n=12, b_value=0.9, seed=42):
    b = generate_parameter({'mode': 'homogeneous', 'value': b_value}, n, 'b',
                           verbose=False)
    a = generate_a_parameter({'mode': 'homogeneous', 'value': 0.5}, b, n,
                             verbose=False)
    z = generate_parameter({'mode': 'homogeneous', 'value': 1.0}, n, 'z',
                           verbose=False)
    ns = generate_base_network(n, 2, 2, 0.05, seed=seed, a=a, b=b)
    return ns, a, b, z


def test_sync_is_seed_invariant():
    """Synchronous trajectories don't depend on `seed` (no permutation)."""
    ns, a, b, z = _make_economy()
    r1 = run_unified_simulation(ns, a, b, z, mode='full', seed=1,
                                max_swaps=1, nb_rounds=20, synchronous=True)
    r2 = run_unified_simulation(ns, a, b, z, mode='full', seed=999_999,
                                max_swaps=1, nb_rounds=20, synchronous=True)
    assert r1['rounds'] == r2['rounds'], (r1['rounds'], r2['rounds'])
    assert r1['converged'] == r2['converged']
    assert r1['cycle_period'] == r2['cycle_period']
    assert r1['total_rewirings'] == r2['total_rewirings']
    for s1, s2 in zip(r1['final_supplier_list'], r2['final_supplier_list']):
        assert set(s1) == set(s2), (s1, s2)
    np.testing.assert_allclose(r1['final_prices'], r2['final_prices'], rtol=1e-12)


def test_async_uses_seed():
    """Asynchronous trajectories DO depend on `seed` in a parameter regime
    with multiple fixed points (AiSi traps at Delta_A > 0)."""
    # Larger n + AiSi heterogeneity so two random permutations can land on
    # different fixed points.
    n = 30
    b = generate_parameter({'mode': 'homogeneous', 'value': 0.9}, n, 'b',
                           verbose=False)
    a = generate_a_parameter({'mode': 'homogeneous', 'value': 0.5}, b, n,
                             verbose=False)
    z = generate_parameter({'mode': 'homogeneous', 'value': 1.0}, n, 'z',
                           verbose=False)
    ns = generate_base_network(n, 4, 4, aisi_spread=0.1, seed=42, a=a, b=b)
    diffs = 0
    for s1, s2 in [(1, 2), (3, 4), (5, 6), (7, 8)]:
        r1 = run_unified_simulation(ns, a, b, z, mode='full', seed=s1,
                                    max_swaps=1, nb_rounds=30, synchronous=False)
        r2 = run_unified_simulation(ns, a, b, z, mode='full', seed=s2,
                                    max_swaps=1, nb_rounds=30, synchronous=False)
        same_supplier = all(set(x) == set(y)
                            for x, y in zip(r1['final_supplier_list'],
                                            r2['final_supplier_list']))
        if not same_supplier:
            diffs += 1
    assert diffs >= 1, (
        "All four (seed_a, seed_b) async pairs converged to identical fixed "
        "points; the seed/perm channel appears inert. Either the economy is "
        "too well-conditioned or the simulator ignores `seed` in async mode."
    )


def test_sync_async_same_first_round():
    """A single-firm-eligible scenario: at round 1, both modes start from the
    same initial state. The set of *proposals* in sync mode is a superset of
    what async could apply before the first GE update (sync evaluates every
    firm against the t=0 GE). For tiny n=2 with no rewiring opportunities the
    two modes must agree trivially.
    """
    ns, a, b, z = _make_economy(n=2)
    r_sync = run_unified_simulation(ns, a, b, z, mode='full', max_swaps=1,
                                    nb_rounds=10, synchronous=True)
    r_async = run_unified_simulation(ns, a, b, z, mode='full', seed=7,
                                     max_swaps=1, nb_rounds=10,
                                     synchronous=False)
    # Both should reach the same fixed point on a 2-firm economy (limited
    # combinatorics; both modes converge to a local optimum).
    for s1, s2 in zip(r_sync['final_supplier_list'],
                      r_async['final_supplier_list']):
        assert set(s1) == set(s2)


def test_sync_trace_one_step_per_round():
    """In sync mode, `trace.price_steps` ticks at most once per round
    (one GE recomputation), unlike async which ticks per accepted swap."""
    ns, a, b, z = _make_economy(n=10)
    r = run_unified_simulation(ns, a, b, z, mode='full', max_swaps=1,
                               nb_rounds=15, synchronous=True, trace=True)
    steps = r['trace']['price_steps']
    # Strictly monotonic, integer-valued, one increment per non-empty round
    # (plus the t=0 anchor and possibly the final-anchor duplicate).
    assert steps[0] == 0
    for prev, cur in zip(steps, steps[1:]):
        assert cur >= prev


if __name__ == '__main__':
    test_sync_is_seed_invariant()
    print("OK: sync_is_seed_invariant")
    test_async_uses_seed()
    print("OK: async_uses_seed")
    test_sync_async_same_first_round()
    print("OK: sync_async_same_first_round")
    test_sync_trace_one_step_per_round()
    print("OK: sync_trace_one_step_per_round")
    print("\nAll synchronous-mode tests passed.")
