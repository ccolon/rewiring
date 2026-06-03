"""
Unit tests for compute_equilibrium_profitmax (profit-maximisation GE, appendix
"Profit maximisation").

Three sanity checks, each on a small random economy:

1. CRS limit (b=1, uniform weights => alpha=1) => the profit-max GE matches
   compute_equilibrium_full element-wise. This is the algebraic equivalence
   that the appendix asserts: the price constant b^{-b*alpha} reduces to 1,
   profits collapse to zero, and B reduces to L.

2. DRS labour-clearing & budget identities. Solve the profit-max GE at b=0.9
   and verify
       L = sum_i a_i b_i v_i  =  n   (chosen labour normalisation)
       B = L + sum_i v_i (1 - b_i alpha_i)
       pi_i = v_i (1 - b_i alpha_i)  >  0       (DRS => 1 - b*alpha > 0)

3. DRS price-equation residual. Plug the returned prices back into Eq. (1)
   and verify the residual is ~0:
       log p_i + log z_i + b_i alpha_i log b_i - (1 - b_i alpha_i) log v_i
       - b_i (1-a_i) sum_j W_ji log p_j  ==  0

Run:
    python tests/test_profitmax_equilibrium.py
"""
import os
import random
import sys

import numpy as np

# Allow `python tests/test_profitmax_equilibrium.py` from any cwd.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from rewiring.equilibrium import (
    compute_adjusted_z,
    compute_equilibrium_full,
    compute_equilibrium_profitmax,
    get_alpha,
)
from rewiring.networks import generate_base_network
from rewiring.parameters import generate_a_parameter, generate_parameter


TOL = 1e-9


def setup_economy(N=10, c=4, cc=2, AiSi_spread=0.0, b_value=1.0, seed=42):
    """Build a small economy: parameters + network + adjusted z + W."""
    random.seed(seed)
    np.random.seed(seed)
    b = generate_parameter({"mode": "homogeneous", "value": b_value}, N, "b", verbose=False)
    a = generate_a_parameter({"mode": "homogeneous", "value": 0.5}, b, N, verbose=False)
    z = generate_parameter({"mode": "homogeneous", "value": 1.0}, N, "z", verbose=False)
    state = generate_base_network(N, c, cc, AiSi_spread, seed=seed,
                                  a=a, b=b, sigma_w=0.0)
    W = state["W0"].copy()
    adjusted_z = compute_adjusted_z(state["AiSi"], state["supplier_id_list"], z)
    return {
        "a": a, "b": b, "z": z, "N": N,
        "state": state, "W": W,
        "adjusted_z": adjusted_z,
    }


# =============================================================================
# TEST 1: CRS limit (b=1, alpha=1) => profitmax matches the baseline
# =============================================================================
def test_crs_limit_matches_baseline():
    # sigma_w=0 with full c-supplier sets => column sums of W = 1 => alpha = 1
    econ = setup_economy(b_value=1.0, seed=42)
    a, b, W, z = econ["a"], econ["b"], econ["W"], econ["adjusted_z"]
    N = econ["N"]

    alpha = get_alpha(a, W)
    assert np.allclose(alpha, 1.0, atol=1e-12), \
        f"this test requires alpha=1; got max|alpha-1|={np.abs(alpha-1).max():.2e}"

    eq_base = compute_equilibrium_full(a, b, z, W, N)
    eq_pm   = compute_equilibrium_profitmax(a, b, z, W, N)

    max_dP = float(np.max(np.abs(eq_base["P"] - eq_pm["P"])))
    max_dX = float(np.max(np.abs(eq_base["X"] - eq_pm["X"])))
    max_pi = float(np.max(np.abs(eq_pm["pi"])))
    print(f"[1] CRS limit: max|dP|={max_dP:.2e}, max|dX|={max_dX:.2e}, "
          f"max|pi|={max_pi:.2e}")
    assert max_dP < TOL, f"prices should match in CRS limit (max dP={max_dP:.2e})"
    assert max_dX < TOL, f"quantities should match in CRS limit (max dX={max_dX:.2e})"
    assert max_pi < TOL, f"profits should vanish in CRS limit (max pi={max_pi:.2e})"


# =============================================================================
# TEST 2: DRS labour-clearing and budget-identity checks
# =============================================================================
def test_drs_clearing_identities():
    econ = setup_economy(b_value=0.9, seed=21)
    a, b, W, z = econ["a"], econ["b"], econ["W"], econ["adjusted_z"]
    N = econ["N"]

    eq = compute_equilibrium_profitmax(a, b, z, W, N)
    v, pi = eq["v"], eq["pi"]

    alpha = get_alpha(a, W)
    bma = b * alpha
    L = float(N)                                # the function defaults to L = n

    # (i) labour clearing
    L_check = float(np.sum(a * b * v))
    # (ii) budget identity B = L + sum_i v_i (1 - b alpha)
    B_check = L + float(np.sum(v * (1 - bma)))
    # (iii) profits match v * (1 - b alpha) and are strictly positive under DRS
    pi_check = v * (1 - bma)

    # (iv) sales linear system  v_i  =  B/n + sum_j (1-a_j) b_j W_ij v_j
    sales_lhs = v
    sales_rhs = (B_check / N) + W @ ((1 - a) * b * v)
    sales_resid = float(np.max(np.abs(sales_lhs - sales_rhs)))

    print(f"[2] DRS clearing: L_check={L_check:.10f} (target {L})  "
          f"B={B_check:.10f}  max|pi_resid|={float(np.max(np.abs(pi-pi_check))):.2e}  "
          f"max|sales_resid|={sales_resid:.2e}  min(pi)={float(pi.min()):.4f}")

    assert abs(L_check - L) < TOL, f"labour clearing off by {L_check - L:.2e}"
    assert np.max(np.abs(pi - pi_check)) < TOL, "pi != v * (1 - b*alpha)"
    assert sales_resid < TOL, f"sales-equation residual = {sales_resid:.2e}"
    assert float(pi.min()) > 0, \
        f"profits should be strictly positive under DRS (min={float(pi.min())})"


# =============================================================================
# TEST 3: DRS price-equation residual
# =============================================================================
def test_drs_price_residual():
    econ = setup_economy(b_value=0.9, seed=21)
    a, b, W, z = econ["a"], econ["b"], econ["W"], econ["adjusted_z"]
    N = econ["N"]

    eq = compute_equilibrium_profitmax(a, b, z, W, N)
    P, v = eq["P"], eq["v"]

    alpha = get_alpha(a, W)
    log_p = np.log(P)

    # Eq. (1) in log form:
    # log p_i = -log z_i - b_i alpha_i log b_i + (1 - b_i alpha_i) log v_i
    #           + b_i (1-a_i) sum_j W_ji log p_j
    rhs = (-np.log(z) - b * alpha * np.log(b) + (1 - b * alpha) * np.log(v)
           + (b * (1 - a)) * (W.T @ log_p))
    resid = float(np.max(np.abs(log_p - rhs)))

    print(f"[3] DRS price residual (log scale): max|resid|={resid:.2e}")
    assert resid < TOL, f"price-equation residual = {resid:.2e}"


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    test_crs_limit_matches_baseline()
    test_drs_clearing_identities()
    test_drs_price_residual()
    print("\nAll tests passed.")
