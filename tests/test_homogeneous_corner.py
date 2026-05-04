"""
Characterisation tests for the pure-homogeneous corner of the parameter space:
    z = 1, a = 0.5, aisi_spread = 0, sigma_w = 0, mode = "full".

We probe three b-regimes and lock in the empirical behaviour:

1. b = 1.0  (CRS + alpha = 1):  the price equation collapses to log p = 0
   (every firm has price 1), so no swap ever reduces cost. The simulation
   is therefore degenerate -- it exits at round 1 with zero rewires. The
   "diversity" reported under dif_init in this corner is purely an artefact
   of initial-network preservation.

2. b = 0.9  (DRS):  prices vary across firms via network position alone.
   Some swaps change the firm's cost, so the simulation rewires.

3. b = 1.1  (IRS):  same as DRS structurally; rewiring expected.

The numbers below are not theorems -- they're the values the model produces
at this seed/n combination, and any code change that breaks them should be
audited.

Run:
    python tests/test_homogeneous_corner.py
"""
import os
import random
import sys

import numpy as np

# Allow `python tests/test_homogeneous_corner.py` from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from rewiring.networks import generate_base_network
from rewiring.simulation import run_unified_simulation


# Test parameters: small enough for fast CI, large enough for non-trivial topology.
N = 50
C = 4
CC = 4
SEED = 0


def _run_pure_homogeneous(b_value, nb_rounds=200, max_swaps=1):
    """Run a single full-mode simulation at the pure-homogeneous corner."""
    random.seed(SEED)
    np.random.seed(SEED)
    a = np.full(N, 0.5)
    b = np.full(N, b_value)
    z = np.full(N, 1.0)
    state = generate_base_network(N, C, CC, aisi_spread=0.0,
                                   seed=SEED, a=a, b=b, sigma_w=0.0)
    return run_unified_simulation(
        state, a, b, z, mode="full",
        seed=SEED, max_swaps=max_swaps, nb_rounds=nb_rounds,
    )


def test_b_eq_one_is_degenerate():
    """At b=1.0, no swap is ever profitable. Algorithm exits round 1 with zero rewires."""
    res = _run_pure_homogeneous(b_value=1.0)
    assert res["converged"] is True, \
        f"b=1.0 should converge in round 1; got converged={res['converged']}"
    assert res["rounds"] == 1, \
        f"b=1.0 should exit at round 1; got rounds={res['rounds']}"
    assert res["total_rewirings"] == 0, \
        f"b=1.0 should produce zero rewires; got total_rewirings={res['total_rewirings']}"
    assert res["cycle_period"] == 1, \
        f"b=1.0 should report cycle_period=1 (strict convergence); got {res['cycle_period']}"
    print(f"  [PASS] b=1.0  rounds={res['rounds']}  rewires={res['total_rewirings']}  "
          f"converged={res['converged']}")


def test_b_below_one_does_rewire():
    """At b=0.9, network-position-induced price asymmetries make some swaps profitable."""
    res = _run_pure_homogeneous(b_value=0.9)
    assert res["total_rewirings"] > 0, \
        f"b=0.9 should produce some rewires; got total_rewirings={res['total_rewirings']}"
    # Cycle_period should be set (system should resolve to either fixed point or period-k cycle,
    # not run out of nb_rounds budget).
    assert res["cycle_period"] is not None, \
        f"b=0.9 should resolve within nb_rounds; got cycle_period=None (truncated)"
    print(f"  [PASS] b=0.9  rounds={res['rounds']}  rewires={res['total_rewirings']}  "
          f"converged={res['converged']}  cycle_period={res['cycle_period']}")


def test_b_above_one_does_rewire():
    """At b=1.1, same as DRS: prices vary by network position, some swaps profitable."""
    res = _run_pure_homogeneous(b_value=1.1)
    assert res["total_rewirings"] > 0, \
        f"b=1.1 should produce some rewires; got total_rewirings={res['total_rewirings']}"
    assert res["cycle_period"] is not None, \
        f"b=1.1 should resolve within nb_rounds; got cycle_period=None (truncated)"
    print(f"  [PASS] b=1.1  rounds={res['rounds']}  rewires={res['total_rewirings']}  "
          f"converged={res['converged']}  cycle_period={res['cycle_period']}")


def main():
    print("=" * 70)
    print(f"Pure-homogeneous corner: N={N}, c={C}, cc={CC}, seed={SEED}")
    print(f"  parameters: a=0.5, z=1.0, aisi_spread=0, sigma_w=0, mode='full'")
    print("=" * 70)
    test_b_eq_one_is_degenerate()
    test_b_below_one_does_rewire()
    test_b_above_one_does_rewire()
    print("=" * 70)
    print("All tests passed.")


if __name__ == "__main__":
    main()
