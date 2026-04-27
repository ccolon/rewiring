"""Tier-limited partial equilibrium.

Public API:
    identify_firms_within_tier_np         -- BFS up to depth `tier` from a focal firm
    compute_partial_equilibrium_cost      -- boundary-conditioned partial GE (manuscript)
    compute_partial_equilibrium_cost_naive -- legacy island-only partial GE
"""

import numpy as np

from .equilibrium import compute_equilibrium_full


def identify_firms_within_tier_np(M: np.ndarray, id_firm: int, tier: int) -> list:
    """BFS of depth `tier` from id_firm on the undirected adjacency (M | M.T).

    Returns the sorted list of firm indices within `tier` hops, including id_firm.
    Matches igraph's neighborhood(..., mode='all') behaviour.
    """
    n = M.shape[0]
    if tier <= 0:
        return [int(id_firm)]
    adj = (M != 0) | (M.T != 0)
    visited = np.zeros(n, dtype=bool)
    visited[id_firm] = True
    frontier = [int(id_firm)]
    for _ in range(int(tier)):
        next_frontier = []
        for u in frontier:
            neighbors = np.where(adj[u] & ~visited)[0]
            for v in neighbors:
                visited[v] = True
                next_frontier.append(int(v))
        if not next_frontier:
            break
        frontier = next_frontier
    return sorted(int(i) for i in np.where(visited)[0])


def compute_partial_equilibrium_cost_naive(a: np.ndarray, b: np.ndarray,
                                           adjusted_z: np.ndarray, W: np.ndarray,
                                           firms_within_tiers: list,
                                           id_rewiring_firm: int) -> float:
    """Naive partial equilibrium: drop all links between in-tier firms and the rest of
    the network, then re-solve `compute_equilibrium_full` on the in-tier sub-economy.

    Legacy behaviour (matches the original `functions.py:compute_partial_equilibrium_and_cost`).
    For the manuscript-faithful boundary-conditioned variant use
    `compute_partial_equilibrium_cost`.
    """
    idx = list(firms_within_tiers)
    W_sub = W[np.ix_(idx, idx)]
    a_sub = a[idx]
    b_sub = b[idx]
    z_sub = adjusted_z[idx]
    n_sub = len(idx)
    partial_eq = compute_equilibrium_full(a_sub, b_sub, z_sub, W_sub, n_sub)
    firm_id_reduced = idx.index(int(id_rewiring_firm))
    return float(partial_eq['P'][firm_id_reduced])


def compute_partial_equilibrium_cost(a: np.ndarray, b: np.ndarray,
                                     adjusted_z_test: np.ndarray,
                                     W_full: np.ndarray, W_test: np.ndarray,
                                     eq_current: dict,
                                     firms_within_tiers: list,
                                     id_rewiring_firm: int,
                                     n_full: int) -> float:
    """Boundary-conditioned partial equilibrium per the manuscript.

    In-tier firms re-equilibrate jointly under the candidate W; outside-firm
    intermediate demand and supplier prices are held fixed at their current values
    (boundary terms phi and psi).

    Algebraic match: at tier >= diameter, recovers `compute_equilibrium_full`.
    Under CRS (b_i = 1) and column sums of W = 1 (alpha_i = 1), at tier = 0 the
    formula collapses to the AA closed-form
        K^AA_i = prod_j P_{j,t}^{(1-a_i) w_{ji}^{(r)}} / z_i^{(r)}.
    """
    idx = list(firms_within_tiers)
    n_in = len(idx)
    a_in = a[idx]
    b_in = b[idx]
    z_in_test = adjusted_z_test[idx]

    alpha_full = a + (1 - a) * W_full.sum(axis=0)
    alpha_test = a + (1 - a) * W_test.sum(axis=0)
    alpha_in = alpha_test[idx]

    P_cur = eq_current['P']
    V_cur = eq_current['X'] * eq_current['P']
    log_P_cur = np.log(P_cur)

    # Sales block
    Wtilde_full_in = W_full[np.ix_(idx, idx)] * ((1 - a_in) / alpha_full[idx])[np.newaxis, :]
    Wtilde_test_in = W_test[np.ix_(idx, idx)] * ((1 - a_in) / alpha_in)[np.newaxis, :]
    in_tier_contribution_cur = Wtilde_full_in @ V_cur[idx]
    phi = V_cur[idx] - 1.0 - in_tier_contribution_cur
    V_in = np.linalg.solve(np.eye(n_in) - Wtilde_test_in, 1.0 + phi)

    # Price block
    factor_row = b_in * (1 - a_in)
    log_factor_total = factor_row * (W_test[:, idx].T @ log_P_cur)
    log_factor_in_tier = factor_row * (W_test[np.ix_(idx, idx)].T @ log_P_cur[idx])
    log_psi = log_factor_total - log_factor_in_tier

    Omega = (-np.log(z_in_test)
             + b_in * alpha_in * np.log(alpha_in)
             + (1 - b_in * alpha_in) * np.log(V_in))

    A = np.eye(n_in) - factor_row[:, np.newaxis] * W_test[np.ix_(idx, idx)].T
    log_P_in = np.linalg.solve(A, Omega + log_psi)

    firm_id_reduced = idx.index(int(id_rewiring_firm))
    return float(np.exp(log_P_in[firm_id_reduced]))
