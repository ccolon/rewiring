"""
Unit tests for compute_partial_equilibrium_cost (boundary-corrected partial GE).

Three sanity checks, each on a small random economy:

1. tier >= diameter  =>  matches compute_equilibrium_full evaluated at the
   candidate W_test (i.e., the 'full' anticipation cost).

2. tier = 0, b = 1, alpha = 1 (CRS + column sums of W = 1)
   =>  matches the AA closed-form  K^AA_i = prod_j P_j^{(1-a_i) w_{ji}^{(r)}} / z_i^{(r)}.

3. tier = 0, general (b != 1 or alpha != 1)
   =>  matches the analytic closed-form
       log P_i^{lim,0} = -log z_i^{(r)} + b_i alpha_i log alpha_i
                        + (1 - b_i alpha_i) log V_i^{cur}
                        + sum_j b_i (1-a_i) w_{ji}^{(r)} log P_j^{cur}.

Run:
    python test_partial_equilibrium.py
"""
import random

import numpy as np

from test_convergence import generate_base_network
from utils import (
    EPSILON,
    build_W_from_suppliers,
    compute_adjusted_z,
    compute_equilibrium_full,
    compute_partial_equilibrium_cost,
    generate_a_parameter,
    generate_parameter,
    identify_firms_within_tier_np,
)


TOL = 1e-9


def setup_economy(N=10, c=4, cc=2, AiSi_spread=0.1, b_value=1.0, seed=42):
    """Build a small economy: parameters + network + current GE."""
    random.seed(seed)
    np.random.seed(seed)
    b = generate_parameter({"mode": "homogeneous", "value": b_value}, N, "b", verbose=False)
    a = generate_a_parameter({"mode": "homogeneous", "value": 0.5}, b, N, verbose=False)
    z = generate_parameter({"mode": "homogeneous", "value": 1.0}, N, "z", verbose=False)
    state = generate_base_network(N, c, cc, AiSi_spread, seed=seed,
                                  a=a, b=b, sigma_w=0.0)
    W = state["W0"].copy()
    adjusted_z = compute_adjusted_z(state["AiSi"], state["supplier_id_list"], z)
    eq = compute_equilibrium_full(a, b, adjusted_z, W, N)
    return {
        "a": a, "b": b, "z": z, "N": N,
        "state": state, "W": W,
        "adjusted_z": adjusted_z, "eq": eq,
    }


def make_candidate_swap(econ, id_firm):
    """Build (W_test, candidate_set, tmp_adjusted_z) for one feasible swap of id_firm."""
    sup = econ["state"]["supplier_id_list"][id_firm]
    alt = econ["state"]["alternate_supplier_id_list"][id_firm]
    if not alt or not sup:
        raise RuntimeError("Firm has no swap to evaluate")
    old_s, new_s = sup[0], alt[0]
    candidate_set = sorted(set(sup) - {old_s} | {new_s})

    W_test = econ["W"].copy()
    W_col = np.zeros(econ["N"])
    Wbar = econ["state"]["Wbar"]
    for s in candidate_set:
        W_col[s] = Wbar[s, id_firm]
    W_test[:, id_firm] = W_col

    tmp_supplier_list = [list(s) for s in econ["state"]["supplier_id_list"]]
    tmp_supplier_list[id_firm] = candidate_set
    tmp_adjusted_z = compute_adjusted_z(econ["state"]["AiSi"], tmp_supplier_list, econ["z"])

    return {
        "W_test": W_test,
        "candidate_set": candidate_set,
        "tmp_adjusted_z": tmp_adjusted_z,
        "id_firm": id_firm,
    }


# =============================================================================
# TEST 1: tier >= diameter  ==>  matches full anticipation
# =============================================================================
def test_tier_full_matches_full_anticipation():
    econ = setup_economy(b_value=0.95, seed=42)
    cand = make_candidate_swap(econ, id_firm=2)

    tier = econ["N"]  # well above diameter

    # Full-network neighborhood at tier=N: every firm
    M_test = (cand["W_test"] != 0).astype(np.int8)
    firms_within_tiers = identify_firms_within_tier_np(M_test, cand["id_firm"], tier)
    # Sanity: should cover everyone if the graph is connected (here it is)
    assert sorted(firms_within_tiers) == list(range(econ["N"])), \
        "tier neighborhood should cover all firms at tier=N"

    cost_lim = compute_partial_equilibrium_cost(
        econ["a"], econ["b"], cand["tmp_adjusted_z"],
        econ["W"], cand["W_test"], econ["eq"],
        firms_within_tiers, cand["id_firm"], econ["N"],
    )
    new_eq = compute_equilibrium_full(
        econ["a"], econ["b"], cand["tmp_adjusted_z"], cand["W_test"], econ["N"],
    )
    cost_full = float(new_eq["P"][cand["id_firm"]])

    print(f"[1] tier=N: limited={cost_lim:.10f}, full={cost_full:.10f}, "
          f"|diff|={abs(cost_lim - cost_full):.2e}")
    assert abs(cost_lim - cost_full) < TOL, \
        f"tier=full should match full anticipation (diff={cost_lim - cost_full:.2e})"


# =============================================================================
# TEST 2: tier=0, b=1, alpha=1 (CRS + col sums = 1)  ==>  matches AA
# =============================================================================
def test_tier0_crs_matches_aa():
    # b = 1.0 (CRS); sigma_w=0 with uniform 1/c weights => column sums = 1 => alpha = 1
    econ = setup_economy(b_value=1.0, seed=7)
    cand = make_candidate_swap(econ, id_firm=3)

    # Verify alpha == 1 on this network (column sums should be 1)
    alpha = econ["a"] + (1 - econ["a"]) * econ["W"].sum(axis=0)
    assert np.allclose(alpha, 1.0, atol=1e-12), \
        f"this test requires alpha=1; got max|alpha-1|={np.abs(alpha-1).max():.2e}"

    tier = 0
    M_test = (cand["W_test"] != 0).astype(np.int8)
    firms_within_tiers = identify_firms_within_tier_np(M_test, cand["id_firm"], tier)
    assert firms_within_tiers == [cand["id_firm"]], \
        "at tier=0 the in-tier set should be just the rewiring firm"

    cost_lim = compute_partial_equilibrium_cost(
        econ["a"], econ["b"], cand["tmp_adjusted_z"],
        econ["W"], cand["W_test"], econ["eq"],
        firms_within_tiers, cand["id_firm"], econ["N"],
    )

    # AA closed form
    i = cand["id_firm"]
    test_z = cand["tmp_adjusted_z"][i]
    P_cur = econ["eq"]["P"]
    W_col = cand["W_test"][:, i]
    cost_aa = float(np.prod(np.power(P_cur, (1 - econ["a"][i]) * W_col)) / test_z)

    print(f"[2] tier=0, CRS: limited={cost_lim:.10f}, AA={cost_aa:.10f}, "
          f"|diff|={abs(cost_lim - cost_aa):.2e}")
    assert abs(cost_lim - cost_aa) < TOL, \
        f"tier=0 + CRS + alpha=1 should equal AA (diff={cost_lim - cost_aa:.2e})"


# =============================================================================
# TEST 3: tier=0, non-CRS  ==>  matches the analytic closed form
# =============================================================================
def test_tier0_nonCRS_matches_closed_form():
    econ = setup_economy(b_value=0.9, seed=21)
    cand = make_candidate_swap(econ, id_firm=4)

    tier = 0
    M_test = (cand["W_test"] != 0).astype(np.int8)
    firms_within_tiers = identify_firms_within_tier_np(M_test, cand["id_firm"], tier)
    assert firms_within_tiers == [cand["id_firm"]]

    cost_lim = compute_partial_equilibrium_cost(
        econ["a"], econ["b"], cand["tmp_adjusted_z"],
        econ["W"], cand["W_test"], econ["eq"],
        firms_within_tiers, cand["id_firm"], econ["N"],
    )

    # Analytic closed form for the boundary-corrected partial eq at tier=0:
    #   log P_i^lim = -log z_i_test
    #              + b_i alpha_i log alpha_i
    #              + (1 - b_i alpha_i) log V_cur[i]
    #              + sum_j b_i (1 - a_i) w_{ji}^{(r)} log P_cur[j]
    # where alpha_i is computed at the candidate W column (= current here since
    # column sums don't change when sigma_w=0 and only one firm's column is touched
    # but the relation alpha_i = a_i + (1-a_i)*sum(W_test[:,i]) holds either way).
    i = cand["id_firm"]
    a_i, b_i = econ["a"][i], econ["b"][i]
    z_i_test = cand["tmp_adjusted_z"][i]
    W_col = cand["W_test"][:, i]
    alpha_i = a_i + (1 - a_i) * W_col.sum()
    V_cur_i = float(econ["eq"]["X"][i] * econ["eq"]["P"][i])
    log_P_cur = np.log(econ["eq"]["P"])

    log_cost_expected = (
        -np.log(z_i_test)
        + b_i * alpha_i * np.log(alpha_i)
        + (1 - b_i * alpha_i) * np.log(V_cur_i)
        + np.sum(b_i * (1 - a_i) * W_col * log_P_cur)
    )
    cost_expected = float(np.exp(log_cost_expected))

    print(f"[3] tier=0, non-CRS (b={b_i:.2f}): limited={cost_lim:.10f}, "
          f"closed_form={cost_expected:.10f}, |diff|={abs(cost_lim - cost_expected):.2e}")
    assert abs(cost_lim - cost_expected) < TOL, \
        f"tier=0 boundary-corrected should match closed form (diff={cost_lim - cost_expected:.2e})"


# =============================================================================
# RUN
# =============================================================================
if __name__ == "__main__":
    test_tier_full_matches_full_anticipation()
    test_tier0_crs_matches_aa()
    test_tier0_nonCRS_matches_closed_form()
    print("\nAll tests passed.")
