"""
Shared utilities for network rewiring simulations.
Contains network generation, equilibrium computation, and parameter generation functions.
"""

import random
from itertools import combinations

import numpy as np
from scipy.sparse.linalg import eigs


# =============================================================================
# CONSTANTS
# =============================================================================

EPSILON = 1e-10


# =============================================================================
# PARAMETER GENERATION
# =============================================================================

def draw_random_vector_normal(mean, sigma, n, bound_min=None, bound_max=None):
    """Draw n values from normal distribution, clipped to bounds."""
    values = np.random.normal(mean, sigma, n)
    if bound_min is not None:
        values = np.maximum(values, bound_min)
    if bound_max is not None:
        values = np.minimum(values, bound_max)
    return values


def generate_parameter(config, n, param_name='param', verbose=True):
    """
    Generate parameter vector based on mode configuration.

    Args:
        config: Dict with 'mode' ('homogeneous', 'uniform', 'normal') and mode-specific params
        n: Number of values to generate
        param_name: Name for logging
        verbose: Print generation info
    """
    mode = config.get('mode', 'homogeneous')

    if mode == 'homogeneous':
        value = config.get('value', 1.0)
        values = np.full(n, value)
        if verbose:
            print(f"  {param_name} (homogeneous): value={value}")

    elif mode == 'uniform':
        min_val = config.get('min', 0.0)
        max_val = config.get('max', 1.0)
        values = np.random.uniform(min_val, max_val, n)
        if verbose:
            print(f"  {param_name} (uniform): [{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")

    elif mode == 'normal':
        mean = config.get('mean', 1.0)
        sigma = config.get('sigma', 0.1)
        bound_min = config.get('bound_min', None)
        bound_max = config.get('bound_max', None)
        values = draw_random_vector_normal(mean, sigma, n, bound_min, bound_max)
        if verbose:
            print(f"  {param_name} (normal): [{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")

    else:
        raise ValueError(f"Unknown mode '{mode}' for parameter {param_name}")

    return values


def generate_a_parameter(a_config, b, n, verbose=True):
    """
    Generate labor share 'a' with constraint that a*b < 1.
    For each firm, max_a = min((1-eps)/b[i], 1-eps).
    """
    mode = a_config.get('mode', 'homogeneous')
    eps = 0.05

    if mode == 'homogeneous':
        value = a_config.get('value', 0.5)
        values = np.minimum(value, (1 - eps) / b)
        if verbose:
            print(f"  a (homogeneous): value={value}, adjusted range=[{values.min():.4f}, {values.max():.4f}]")

    elif mode == 'uniform':
        min_val = a_config.get('min', 0.1)
        max_vals = np.minimum(a_config.get('max', 0.9), (1 - eps) / b)
        values = np.array([np.random.uniform(min_val, max_val) for max_val in max_vals])
        if verbose:
            print(f"  a (uniform): [{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")

    elif mode == 'normal':
        mean = a_config.get('mean', 0.5)
        sigma = a_config.get('sigma', 0.1)
        min_a = eps
        max_vals = np.minimum((1 - eps) / b, 1 - eps)
        values = np.array([
            draw_random_vector_normal(mean, sigma, 1, min_a, max_val)[0]
            for max_val in max_vals
        ])
        if verbose:
            print(f"  a (normal): [{values.min():.4f}, {values.max():.4f}], mean={values.mean():.4f}")

    else:
        raise ValueError(f"Unknown mode '{mode}' for parameter a")

    return values


# =============================================================================
# NETWORK GENERATION (pure numpy, SF-FA topology)
# =============================================================================

def initialize_sffa_adjacency(n: int, c: int) -> np.ndarray:
    """
    Create initial adjacency matrix using Scale-Free Fitness-Attraction model.
    Returns M where M[supplier, customer] = 1 if edge exists.
    """
    exp_in = 1.35 + 1
    exp_out = 1.26 + 1
    i0 = 12

    fitness_in = np.power(np.arange(1, n + 1) + i0 - 1, -1 / (exp_in - 1))
    fitness_out = np.power(np.arange(1, n + 1) + i0 - 1, -1 / (exp_out - 1))
    fitness_in = fitness_in / fitness_in.sum()
    fitness_out = fitness_out / fitness_out.sum()

    num_edges = n * (c - 1)
    M = np.zeros((n, n), dtype=np.float64)
    edges_created = 0

    for _ in range(num_edges * 100):
        if edges_created >= num_edges:
            break
        source = np.random.choice(n, p=fitness_out)
        target = np.random.choice(n, p=fitness_in)
        if source != target and M[source, target] == 0:
            M[source, target] = 1
            edges_created += 1

    return M


def add_one_supplier_each_firm(M: np.ndarray) -> np.ndarray:
    """Ensure every firm has at least one supplier."""
    n = M.shape[0]
    for i in range(n):
        potential_suppliers = [j for j in range(n) if j != i and M[j, i] == 0]
        if potential_suppliers:
            new_supplier = random.choice(potential_suppliers)
            M[new_supplier, i] = 1
    return M


def identify_suppliers_from_matrix(M: np.ndarray):
    """Extract supplier lists from adjacency matrix."""
    n = M.shape[0]
    supplier_id_list = [list(np.where(M[:, i] == 1)[0]) for i in range(n)]
    nb_suppliers = np.array([len(s) for s in supplier_id_list])
    return supplier_id_list, nb_suppliers


def create_technology_matrix(M: np.ndarray, nb_suppliers: np.ndarray,
                             nb_extra_suppliers: np.ndarray, supplier_id_list: list,
                             a: np.ndarray = None, b: np.ndarray = None,
                             sigma_w: float = 0.0):
    """
    Create technology matrix Wbar with link weights and alternate suppliers.

    Args:
        M: Adjacency matrix
        nb_suppliers: Number of suppliers per firm
        nb_extra_suppliers: Number of alternate suppliers per firm
        supplier_id_list: Current suppliers for each firm
        a: Labor share array (optional, for weight bounds with heterogeneous params)
        b: Returns to scale array (optional, for weight bounds)
        sigma_w: Weight variance (0 = uniform weights 1/nb_suppliers)
    """
    n = M.shape[0]
    Wbar = np.zeros((n, n), dtype=np.float64)
    alternate_supplier_id_list = []
    eps = 0.05

    for i in range(n):
        current_suppliers = supplier_id_list[i]
        other_firms = [j for j in range(n) if j != i]
        potential_alternates = list(set(other_firms) - set(current_suppliers))

        if len(potential_alternates) <= nb_extra_suppliers[i]:
            alternates = potential_alternates
        else:
            alternates = random.sample(potential_alternates, nb_extra_suppliers[i])

        alternate_supplier_id_list.append(alternates)

        # Compute weight
        mean_weight = 1.0 / nb_suppliers[i] if nb_suppliers[i] > 0 else 1.0

        # Weight bounds (only relevant if a, b provided and sigma_w > 0)
        if a is not None and b is not None:
            min_val = eps
            max_val = (1 - eps) / (b[i] * (1 - a[i])) if b[i] * (1 - a[i]) > eps else 10.0
        else:
            min_val = eps
            max_val = 10.0

        all_suppliers = current_suppliers + alternates
        for supplier in all_suppliers:
            if sigma_w > 0:
                weight = draw_random_vector_normal(mean_weight, sigma_w, 1, min_val, max_val)[0]
            else:
                weight = np.clip(mean_weight, min_val, max_val)
            Wbar[supplier, i] = weight

    return Wbar, alternate_supplier_id_list


def get_AiSi_productivities(supplier_id_list: list, alternate_supplier_id_list: list,
                            spread: float) -> list:
    """Generate supplier-combination-specific productivity multipliers."""
    nb_suppliers_per_firm = [len(s) for s in supplier_id_list]
    all_potential_suppliers_per_firm = [
        sorted(supplier_id_list[i] + alternate_supplier_id_list[i])
        for i in range(len(supplier_id_list))
    ]
    all_combinations_of_suppliers = [
        list(combinations(all_suppliers, c_i))
        for c_i, all_suppliers in zip(nb_suppliers_per_firm, all_potential_suppliers_per_firm)
    ]
    return [
        {combi: random.uniform(1 - spread, 1 + spread) for combi in all_combi}
        for all_combi in all_combinations_of_suppliers
    ]


def compute_adjusted_z(AiSi: list, supplier_id_list: list, z: np.ndarray = None) -> np.ndarray:
    """
    Compute productivity adjusted by supplier-combination-specific multiplier.

    Args:
        AiSi: Supplier-combination productivity multipliers
        supplier_id_list: Current suppliers for each firm
        z: Base productivity array (if None, treated as 1.0 for all firms)
    """
    n = len(AiSi)
    if z is None:
        z = np.ones(n)
    return np.array([
        z[i] * AiSi[i][tuple(sorted(int(s) for s in supplier_id_list[i]))]
        for i in range(n)
    ])


# =============================================================================
# EQUILIBRIUM COMPUTATION
# =============================================================================

def get_alpha(a: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute alpha = a + (1-a) * sum(W, axis=0)."""
    return a + (1 - a) * np.sum(W, axis=0)


def compute_equilibrium_crs(a: float, adjusted_z: np.ndarray, W: np.ndarray, n: int) -> dict:
    """
    Compute economic equilibrium for CRS case (b=1) with alpha=1.

    Simplified version where b=1 and column sums of W equal 1, giving alpha=1.
    Price equation: (I - (1-a)*W.T) * log(p) = -log(adjusted_z)

    Args:
        a: Labor share (scalar, homogeneous)
        adjusted_z: Productivity array (already adjusted by AiSi)
        W: Weight matrix (active links only)
        n: Number of firms
    """
    # Transition matrix for sales
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

    # Normalize: kappa = n / sum(v * a) since alpha=1
    kappa = n / (v_unnormalized * a).sum()
    v = kappa * v_unnormalized

    # Price equation: (I - (1-a)*W.T) * log(p) = -log(adjusted_z)
    A_matrix = np.eye(n) - (1 - a) * W.T
    b_vector = -np.log(adjusted_z)
    log_p = np.linalg.solve(A_matrix, b_vector)
    p = np.exp(log_p)

    return {"X": v / p, "P": p}


def compute_equilibrium_full(a: np.ndarray, b: np.ndarray, z: np.ndarray,
                             W: np.ndarray, n: int) -> dict:
    """
    Compute economic equilibrium with heterogeneous parameters.

    Full version supporting heterogeneous a, b, z arrays.

    Args:
        a: Labor share array
        b: Returns to scale array
        z: Productivity array (already adjusted by AiSi)
        W: Weight matrix (active links only)
        n: Number of firms
    """
    alpha = get_alpha(a, W)

    # Transition matrix for sales
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

    # Normalize
    kappa = n / np.sum(v_unnormalized * a / alpha)
    v = kappa * v_unnormalized

    # Price equation
    b_vector = (-np.log(z) + b * alpha * np.log(alpha) + (1 - b * alpha) * np.log(v))
    A_matrix = np.eye(n) - (b * (1 - a))[:, np.newaxis] * W.T

    log_p = np.linalg.solve(A_matrix, b_vector)
    p = np.exp(log_p)

    return {"X": v / p, "P": p}


def build_W_from_suppliers(supplier_set: list, Wbar: np.ndarray) -> np.ndarray:
    """Build W matrix from supplier sets."""
    n = Wbar.shape[0]
    W = np.zeros((n, n), dtype=Wbar.dtype)
    for buyer, suppliers in enumerate(supplier_set):
        if suppliers:
            W[list(suppliers), buyer] = Wbar[list(suppliers), buyer]
    return W


def calculate_utility(eq: dict) -> float:
    """Calculate utility as negative sum of positive prices."""
    prices = eq['P']
    return -np.sum(prices[prices > 0])


# =============================================================================
# PARTIAL EQUILIBRIUM (LIMITED ANTICIPATION)
# =============================================================================

def identify_firms_within_tier_np(M: np.ndarray, id_firm: int, tier: int) -> list:
    """BFS of depth `tier` from id_firm on undirected adjacency (M | M.T).

    Returns sorted list of firm indices within tier hops (including id_firm).
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
    """Naive partial equilibrium: drop all links between in-tier firms and the
    rest of the network and re-solve `compute_equilibrium_full` on the island.

    This is the legacy behaviour from `functions.py:compute_partial_equilibrium_and_cost`.
    It does not match the boundary-conditioned partial equilibrium described in
    the manuscript -- for that, use `compute_partial_equilibrium_cost`.
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
    intermediate demand and supplier prices are held fixed at their current
    values (boundary terms phi and psi).

    Algebraic match: at tier >= diameter, recovers `compute_equilibrium_full`.
    Under CRS (b_i=1) and column sums of W = 1 (alpha_i=1), at tier=0 the
    formula collapses to the AA closed-form
        K^AA_i = prod_j P_{j,t}^{(1-a_i) w_{ji}^{(r)}} / z_i^{(r)}.

    Args:
        a, b: full-network parameter arrays of length n_full (constant across swap).
        adjusted_z_test: AiSi-adjusted z reflecting the candidate supplier set.
        W_full: current input-output matrix (pre-swap).
        W_test: candidate input-output matrix (post-swap), differs from W_full
                only in the rewiring firm's column.
        eq_current: GE result under (a, b, current adjusted_z, W_full); must
                    contain at least 'P' and 'X' arrays of length n_full.
        firms_within_tiers: indices of firms in the tier neighborhood
                            (must include id_rewiring_firm).
        id_rewiring_firm: index of the firm evaluating the swap.
        n_full: total number of firms in the full network.

    Returns:
        Anticipated price of id_rewiring_firm under the candidate W_test,
        with boundary held fixed at the current equilibrium.
    """
    idx = list(firms_within_tiers)
    n_in = len(idx)
    a_in = a[idx]
    b_in = b[idx]
    z_in_test = adjusted_z_test[idx]

    # Full alpha vectors (depend on column sums; differ between current and candidate
    # only at the rewiring firm's column when its column sum changes)
    alpha_full = a + (1 - a) * W_full.sum(axis=0)
    alpha_test = a + (1 - a) * W_test.sum(axis=0)
    alpha_in = alpha_test[idx]

    # Current GE state (held-fixed boundary)
    P_cur = eq_current['P']
    V_cur = eq_current['X'] * eq_current['P']  # nominal sales per firm
    log_P_cur = np.log(P_cur)

    # ------------------------------------------------------------------
    # Sales block: V_in = Wtilde_test_in @ V_in + (1 + phi)
    # where Wtilde[k,j] = (1 - a_j) / alpha_j * W[k,j]
    # and phi[k] = sum_{j not in T} Wtilde_full[k,j] V_cur[j]
    # computed by subtraction: phi = (V_cur[k] - 1) - in-tier-only-cur-contribution[k]
    # ------------------------------------------------------------------
    Wtilde_full_in = W_full[np.ix_(idx, idx)] * ((1 - a_in) / alpha_full[idx])[np.newaxis, :]
    Wtilde_test_in = W_test[np.ix_(idx, idx)] * ((1 - a_in) / alpha_in)[np.newaxis, :]

    in_tier_contribution_cur = Wtilde_full_in @ V_cur[idx]
    phi = V_cur[idx] - 1.0 - in_tier_contribution_cur

    V_in = np.linalg.solve(np.eye(n_in) - Wtilde_test_in, 1.0 + phi)

    # ------------------------------------------------------------------
    # Price block: log P_in = (b*(1-a))_in W_test_in.T @ log P_in + Omega + log_psi
    # log_psi[k] = sum_{j not in T} b_k(1-a_k) W_test[j,k] log P_cur[j]
    # computed by subtraction
    # ------------------------------------------------------------------
    factor_row = b_in * (1 - a_in)  # shape (n_in,)

    log_factor_total = factor_row * (W_test[:, idx].T @ log_P_cur)        # all suppliers
    log_factor_in_tier = factor_row * (W_test[np.ix_(idx, idx)].T @ log_P_cur[idx])
    log_psi = log_factor_total - log_factor_in_tier

    Omega = (-np.log(z_in_test)
             + b_in * alpha_in * np.log(alpha_in)
             + (1 - b_in * alpha_in) * np.log(V_in))

    A = np.eye(n_in) - factor_row[:, np.newaxis] * W_test[np.ix_(idx, idx)].T
    log_P_in = np.linalg.solve(A, Omega + log_psi)

    firm_id_reduced = idx.index(int(id_rewiring_firm))
    return float(np.exp(log_P_in[firm_id_reduced]))
