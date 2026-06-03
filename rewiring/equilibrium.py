"""General-equilibrium computation.

Public API:
    get_alpha                     -- alpha_i = a_i + (1-a_i) * sum_j W_ji
    compute_equilibrium_crs       -- specialised solver for b=1, alpha=1 (CRS, column sums = 1)
    compute_equilibrium_full      -- general solver for heterogeneous (a, b, z)
                                     (cost-min / contestability baseline)
    compute_equilibrium_profitmax -- profit-maximisation variant (appendix
                                     "Profit maximisation"): sales become a
                                     linear system with profits rebated to
                                     households; price constant changes from
                                     alpha^{b*alpha} to b^{-b*alpha}.
    compute_adjusted_z            -- z_i scaled by the AiSi multiplier of the
                                     active supplier set
    calculate_utility             -- -sum_i p_i over positive prices
    compute_static_gap            -- per-firm "best-attainable cost given current
                                     state" (theta_static query for terminal
                                     configurations)
"""

from itertools import combinations

import numpy as np
from scipy.sparse.linalg import eigs

from .networks import build_W_from_suppliers
from .parameters import EPSILON


def get_alpha(a: np.ndarray, W: np.ndarray) -> np.ndarray:
    """alpha_i = a_i + (1 - a_i) * sum_j W_ji."""
    return a + (1 - a) * np.sum(W, axis=0)


def compute_adjusted_z(AiSi: list, supplier_id_list: list, z: np.ndarray = None) -> np.ndarray:
    """Multiply base z_i by the AiSi multiplier of the firm's active supplier set.

    If z is None, the base productivity is treated as 1.
    """
    n = len(AiSi)
    if z is None:
        z = np.ones(n)
    return np.array([
        z[i] * AiSi[i][tuple(sorted(int(s) for s in supplier_id_list[i]))]
        for i in range(n)
    ])


def compute_equilibrium_crs(a: float, adjusted_z: np.ndarray, W: np.ndarray, n: int) -> dict:
    """Equilibrium for the CRS special case (b=1, alpha=1).

    Sales transition  M = a/n + (1-a) * W
    Price equation    (I - (1-a) W^T) log p = -log z

    `a` is a scalar (homogeneous labor share).
    """
    M = a / n + (1 - a) * W

    if n > 10:
        eigenvalues, eigenvectors = eigs(M, k=1, which='LM')
        if abs(1.0 - eigenvalues[0].real) > EPSILON:
            raise ValueError(f"Eigenvalue is not 1: {eigenvalues[0].real}")
        v_unnormalized = eigenvectors[:, 0].real
    else:
        eigenvalues, eigenvectors = np.linalg.eig(M)
        index = np.isclose(eigenvalues, 1)
        if not index.any():
            raise ValueError('No eigenvalue 1')
        v_unnormalized = eigenvectors[:, index].T.real

    v_unnormalized = np.abs(v_unnormalized).flatten()
    if v_unnormalized.sum() < EPSILON:
        raise ValueError("All sales are null")

    kappa = n / (v_unnormalized * a).sum()
    v = kappa * v_unnormalized

    A_matrix = np.eye(n) - (1 - a) * W.T
    b_vector = -np.log(adjusted_z)
    log_p = np.linalg.solve(A_matrix, b_vector)
    p = np.exp(log_p)

    return {"X": v / p, "P": p}


def compute_equilibrium_full(a: np.ndarray, b: np.ndarray, z: np.ndarray,
                             W: np.ndarray, n: int) -> dict:
    """General equilibrium with heterogeneous (a, b, z) arrays."""
    alpha = get_alpha(a, W)
    M = (1 / alpha) * (a / n + (1 - a)[np.newaxis, :] * W)

    if n > 10:
        eigenvalues, eigenvectors = eigs(M, k=1, which='LM')
        if abs(1.0 - eigenvalues[0].real) > EPSILON:
            raise ValueError(f"Eigenvalue is not 1: {eigenvalues[0].real}")
        v_unnormalized = eigenvectors[:, 0].real
    else:
        eigenvalues, eigenvectors = np.linalg.eig(M)
        index = np.isclose(eigenvalues, 1)
        if not index.any():
            raise ValueError('No eigenvalue 1')
        v_unnormalized = eigenvectors[:, index].T.real

    v_unnormalized = np.abs(v_unnormalized).flatten()
    if v_unnormalized.sum() < EPSILON:
        raise ValueError("All sales are null")

    kappa = n / np.sum(v_unnormalized * a / alpha)
    v = kappa * v_unnormalized

    b_vector = -np.log(z) + b * alpha * np.log(alpha) + (1 - b * alpha) * np.log(v)
    A_matrix = np.eye(n) - (b * (1 - a))[:, np.newaxis] * W.T

    log_p = np.linalg.solve(A_matrix, b_vector)
    p = np.exp(log_p)

    return {"X": v / p, "P": p}


def compute_equilibrium_profitmax(a: np.ndarray, b: np.ndarray, z: np.ndarray,
                                  W: np.ndarray, n: int, L: float = None) -> dict:
    """Profit-maximisation general equilibrium (appendix "Profit maximisation").

    Replaces the cost-minimisation / contestability baseline
    (`compute_equilibrium_full`) with firms that price at marginal cost and
    rebate profits to households. The wage stays as numeraire (h=1), matching
    the baseline.

    Equations (with alpha_i = a_i + (1-a_i) sum_j W_ji = tilde a_i):

        Price :  p_i = (z_i * b_i^{b_i alpha_i})^{-1}
                       * v_i^{1 - b_i alpha_i}
                       * prod_j p_j^{b_i (1-a_i) W_ji}                     (1)
        Sales :  v_i = B/n + sum_j (1 - a_j) b_j W_ij v_j                  (2)
        Budget:  B   = L + sum_i v_i (1 - b_i alpha_i)                     (3)
        Labour:  L   = sum_i a_i b_i v_i                                   (4)
        Profit:  pi_i = v_i (1 - b_i alpha_i)                              (5)

    Differences vs. `compute_equilibrium_full` (only two):
      - Price constant: b_i^{-b_i alpha_i} replaces alpha_i^{b_i alpha_i}
        (i.e. RHS gains  -b*alpha*log(b)  in place of  +b*alpha*log(alpha)).
      - Sales equation is a *linear system* in v rather than a Perron-Frobenius
        eigenproblem, because the budget B now contains rebated profits.

    Algorithm:
      1. Solve (I - W diag((1-a) b)) w_vec = 1.
      2. Pin scale via labour clearing L = sum_i a_i b_i v_i, default L = n
         to match the baseline normalisation (kappa = n / sum a*v/alpha).
         Hence B = n * L / ((a*b) . w_vec)  and  v = (B/n) w_vec.
         The budget identity (3) then holds automatically by Walras' law.
      3. Solve the same log-linear price system as the baseline with the
         modified constant.

    Returns:
        {'X': v/p, 'P': p, 'v': v, 'pi': pi}

    Notes:
      - Under CRS with column sums of W equal to 1 (b_i = 1, alpha_i = 1)
        profits collapse to zero and this function returns the same v, p as
        `compute_equilibrium_full` (modulo floating-point).
      - IRS (b_i alpha_i > 1) makes the profit objective ill-posed (see
        appendix) and is rejected upstream by `run_unified_simulation`'s
        guard, not here.
    """
    if L is None:
        L = float(n)

    alpha = get_alpha(a, W)

    # ----- Sales: linear system (I - W diag((1-a) b)) w_vec = 1, then scale.
    # W * D[newaxis, :] multiplies each column j of W by D_j.
    D = (1.0 - a) * b                                            # shape (n,)
    A_v = np.eye(n) - W * D[np.newaxis, :]
    w_vec = np.linalg.solve(A_v, np.ones(n))

    # Labour clearing pins B:  L = (B/n) * (a*b) . w_vec   =>   B = n*L / (ab @ w).
    ab_w = float(np.dot(a * b, w_vec))
    if abs(ab_w) < EPSILON:
        raise ValueError(
            "Labour-clearing denominator (a*b) . w is ~0; cannot pin B."
        )
    B = n * L / ab_w
    v = (B / n) * w_vec

    # ----- Price equation: same A_matrix as baseline; only the constant changes.
    # Baseline RHS constant:    + b * alpha * log(alpha)
    # Profit-max RHS constant:  - b * alpha * log(b)
    A_matrix = np.eye(n) - (b * (1 - a))[:, np.newaxis] * W.T
    b_vector = -np.log(z) - b * alpha * np.log(b) + (1 - b * alpha) * np.log(v)
    log_p = np.linalg.solve(A_matrix, b_vector)
    p = np.exp(log_p)

    pi = v * (1.0 - b * alpha)

    return {"X": v / p, "P": p, "v": v, "pi": pi}


def calculate_utility(eq: dict) -> float:
    """Utility = -sum of positive prices."""
    prices = eq['P']
    return -np.sum(prices[prices > 0])


def compute_static_gap(base_ns: dict, final_supplier_list: list,
                       a: np.ndarray, b: np.ndarray, z: np.ndarray,
                       static_max_swaps: int):
    """Per-firm "best-attainable cost given the current state" (theta_static query).

    For each firm i at the configuration `final_supplier_list`, enumerate every
    candidate supplier set reachable by up to `static_max_swaps` simultaneous
    swaps from i's current set, and pick the candidate with the lowest
    counterfactual price P[i].

    With `static_max_swaps = min(c, c')` the enumeration covers every size-c
    subset of i's pool (current suppliers + alternates), i.e. the firm-level
    cost frontier under full visibility and unlimited swap. With
    `static_max_swaps = 1` it reproduces the kappa=1-reachable query.

    For each candidate, the full GE is recomputed with only firm i's W-column
    replaced (other firms' supplier sets are held fixed at the terminal state).
    No firm actually acts; this is a static query.

    The firm's current alternates list at the terminal state is reconstructed
    from the union of base_ns['supplier_id_list'][i] (initial suppliers) and
    base_ns['alternate_supplier_id_list'][i] (initial alternates), minus the
    firm's current supplier set -- the simulation only moves entries between
    those two lists.

    Returns:
        p_current   : np.ndarray (n,) -- price under current GE.
        p_best      : np.ndarray (n,) -- min p[i] over enumerated candidates.
        theta_i     : p_current - p_best.
        theta_static: float -- (sum_p_current - sum_p_best) / sum_p_current.
    """
    n = len(final_supplier_list)
    Wbar = base_ns['Wbar']
    AiSi = base_ns['AiSi']
    alt = base_ns['alternate_supplier_id_list']

    sup_now = [list(s) for s in final_supplier_list]
    W_now = build_W_from_suppliers(sup_now, Wbar)
    adj_z_now = compute_adjusted_z(AiSi, sup_now, z)
    eq_now = compute_equilibrium_full(a, b, adj_z_now, W_now, n)
    p_current = np.asarray(eq_now['P'])

    base_pool = [set(base_ns['supplier_id_list'][i]) | set(alt[i])
                 for i in range(n)]
    alt_now = [sorted(base_pool[i] - set(sup_now[i])) for i in range(n)]

    p_best = p_current.copy()

    for i in range(n):
        current_set = set(sup_now[i])
        alternates = alt_now[i]
        for swap_size in range(1, static_max_swaps + 1):
            if len(alternates) < swap_size or len(current_set) < swap_size:
                continue
            for new_sups in combinations(alternates, swap_size):
                for old_sups in combinations(current_set, swap_size):
                    cand_set = (current_set - set(old_sups)) | set(new_sups)

                    W_test = W_now.copy()
                    W_test[:, i] = 0.0
                    for s in cand_set:
                        W_test[s, i] = Wbar[s, i]

                    tmp_sup_i = sorted(int(s) for s in cand_set)
                    tmp_supplier_list = sup_now.copy()
                    tmp_supplier_list[i] = tmp_sup_i
                    adj_z_test = compute_adjusted_z(AiSi, tmp_supplier_list, z)

                    new_eq = compute_equilibrium_full(a, b, adj_z_test,
                                                     W_test, n)
                    cost_i = float(new_eq['P'][i])
                    if cost_i < p_best[i]:
                        p_best[i] = cost_i

    theta_i = p_current - p_best
    sum_pc = float(p_current.sum())
    sum_pb = float(p_best.sum())
    theta_static = (sum_pc - sum_pb) / sum_pc if sum_pc > 0 else 0.0
    return p_current, p_best, theta_i, theta_static
