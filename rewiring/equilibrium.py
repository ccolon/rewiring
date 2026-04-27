"""General-equilibrium computation.

Public API:
    get_alpha                -- alpha_i = a_i + (1-a_i) * sum_j W_ji
    compute_equilibrium_crs  -- specialised solver for b=1, alpha=1 (CRS, column sums = 1)
    compute_equilibrium_full -- general solver for heterogeneous (a, b, z)
    compute_adjusted_z       -- z_i scaled by the AiSi multiplier of the active supplier set
    calculate_utility        -- -sum_i p_i over positive prices
"""

import numpy as np
from scipy.sparse.linalg import eigs

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


def calculate_utility(eq: dict) -> float:
    """Utility = -sum of positive prices."""
    prices = eq['P']
    return -np.sum(prices[prices > 0])
