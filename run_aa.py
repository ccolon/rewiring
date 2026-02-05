"""
Standalone AA (Asynchronous Algorithm) mode simulation for network rewiring.
Independent from run.py and config.yml - all parameters are set in this file.

AA mode uses frozen reference prices within each round, with synchronous updates
and a price ratchet mechanism. Valid under CRS (b=1) with alpha=1.
"""

import random
from datetime import datetime
from itertools import combinations

import numpy as np
from scipy.sparse.linalg import eigs


# =============================================================================
# PARAMETERS (all hardcoded)
# =============================================================================

# Simulation parameters
NB_ROUNDS = 50
NB_FIRMS = 50
C = 4  # base connectivity (edges per node in initial network)
CC = 1  # number of extra potential suppliers per firm

# Economic parameters (homogeneous)
# b=1 (CRS) and weights sum to 1 => alpha = a + (1-a)*1 = 1 always
A_VALUE = 0.5  # labor share

# Network parameters
AISI_SPREAD = 0.2  # AiSi productivity spread parameter

# Numerical parameters
EPSILON = 1e-10
CONVERGENCE_THRESHOLD = 1e-10


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

    # Fitness vectors
    fitness_in = np.power(np.arange(1, n + 1) + i0 - 1, -1 / (exp_in - 1))
    fitness_out = np.power(np.arange(1, n + 1) + i0 - 1, -1 / (exp_out - 1))

    # Normalize to get probabilities
    fitness_in = fitness_in / fitness_in.sum()
    fitness_out = fitness_out / fitness_out.sum()

    # Number of edges to create (c-1 because we'll add one per firm later)
    num_edges = n * (c - 1)

    # Create adjacency matrix
    M = np.zeros((n, n), dtype=np.float64)

    edges_created = 0
    max_attempts = num_edges * 100

    for _ in range(max_attempts):
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
                             nb_extra_suppliers: np.ndarray,
                             supplier_id_list: list):
    """
    Create technology matrix Wbar with link weights and alternate suppliers.
    Weights are 1/nb_suppliers so each column sums to 1.
    """
    n = M.shape[0]
    Wbar = np.zeros((n, n), dtype=np.float64)
    alternate_supplier_id_list = []

    for i in range(n):
        current_suppliers = supplier_id_list[i]
        other_firms = [j for j in range(n) if j != i]
        potential_alternates = list(set(other_firms) - set(current_suppliers))

        if len(potential_alternates) <= nb_extra_suppliers[i]:
            alternates = potential_alternates
        else:
            alternates = random.sample(potential_alternates, nb_extra_suppliers[i])

        alternate_supplier_id_list.append(alternates)

        # Weight = 1/nb_suppliers ensures column sums to 1
        weight = 1.0 / nb_suppliers[i] if nb_suppliers[i] > 0 else 1.0
        for supplier in current_suppliers + alternates:
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


def compute_adjusted_z(AiSi: list, supplier_id_list: list) -> np.ndarray:
    """Compute productivity (= AiSi since base z=1)."""
    return np.array([
        AiSi[i][tuple(sorted(int(s) for s in supplier_id_list[i]))]
        for i in range(len(AiSi))
    ])


# =============================================================================
# EQUILIBRIUM COMPUTATION (simplified for alpha=1, b=1, base z=1)
# =============================================================================

def compute_equilibrium(a: float, adjusted_z: np.ndarray, W: np.ndarray, n: int) -> dict:
    """
    Compute economic equilibrium. Simplified for CRS (b=1) with alpha=1.

    With alpha=1:
    - M = a/n + (1-a)*W
    - Price equation: (I - (1-a)*W.T) * log(p) = -log(adjusted_z)
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


def build_W_from_suppliers(supplier_set: list, Wbar: np.ndarray) -> np.ndarray:
    """Build W matrix from supplier sets."""
    n = Wbar.shape[0]
    W = np.zeros((n, n), dtype=Wbar.dtype)
    for buyer, suppliers in enumerate(supplier_set):
        if suppliers:
            W[list(suppliers), buyer] = Wbar[list(suppliers), buyer]
    return W


# =============================================================================
# MAIN AA SIMULATION
# =============================================================================

def run_aa_simulation():
    """Run the AA mode simulation."""

    print("=" * 60)
    print("AA Mode Simulation (alpha=1 simplified)")
    print("=" * 60)

    starting_time = datetime.now()
    n = NB_FIRMS
    a = A_VALUE

    print(f"\nParameters: n={n}, rounds={NB_ROUNDS}, c={C}, cc={CC}, a={a}, AiSi_spread={AISI_SPREAD}")

    # Generate network
    print("\nGenerating SF-FA network...")
    M0 = add_one_supplier_each_firm(initialize_sffa_adjacency(n, C))
    supplier_id_list, nb_suppliers = identify_suppliers_from_matrix(M0)
    Wbar, alternate_supplier_id_list = create_technology_matrix(
        M0, nb_suppliers, np.full(n, CC), supplier_id_list
    )
    W0 = M0 * Wbar
    AiSi = get_AiSi_productivities(supplier_id_list, alternate_supplier_id_list, AISI_SPREAD)

    print(f"  Edges: {int(M0.sum())}, Mean suppliers: {nb_suppliers.mean():.2f}")

    # Initial equilibrium
    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
    eq = compute_equilibrium(a, adjusted_z, W0, n)
    initial_utility = -eq['P'].sum()

    print(f"  Initial utility: {initial_utility:.4f}, prices: [{eq['P'].min():.4f}, {eq['P'].max():.4f}]")

    # Initialize AA variables
    P_ref = eq['P'].copy()
    aa_best_cost = P_ref.copy()

    # Main loop
    print("\n" + "-" * 60)
    converged = False
    final_round = NB_ROUNDS

    for r in range(1, NB_ROUNDS + 1):
        aa_do = np.zeros(n, dtype=bool)
        aa_remove = [None] * n
        aa_add = [None] * n

        for id_firm in np.random.permutation(n):
            potential_cost = aa_best_cost[id_firm]
            best_remove, best_add = None, None

            # Evaluate all single swaps from CURRENT configuration
            for new_sup in alternate_supplier_id_list[id_firm]:
                for old_sup in supplier_id_list[id_firm]:
                    # Build test W column
                    W_col = np.zeros(n)
                    for s in supplier_id_list[id_firm]:
                        if s != old_sup:
                            W_col[s] = Wbar[s, id_firm]
                    W_col[new_sup] = Wbar[new_sup, id_firm]

                    # Compute adjusted_z for test configuration
                    test_suppliers = sorted(
                        (set(supplier_id_list[id_firm]) | {new_sup}) - {old_sup}
                    )
                    test_z = AiSi[id_firm][tuple(int(s) for s in test_suppliers)]

                    # Cost formula (CRS): prod(P^((1-a)*W)) / z
                    cost = np.prod(np.power(P_ref, (1 - a) * W_col)) / test_z

                    if cost < potential_cost - EPSILON:
                        potential_cost = cost
                        best_remove, best_add = old_sup, new_sup

            if best_add is not None:
                aa_do[id_firm] = True
                aa_remove[id_firm] = best_remove
                aa_add[id_firm] = best_add
                aa_best_cost[id_firm] = potential_cost

        # Update reference prices
        P_next = aa_best_cost.copy()
        max_rel = np.max(np.abs(P_next - P_ref) / np.maximum(P_ref, 1e-12))

        print(f"Round {r:3d}: swaps={aa_do.sum():3d}, max_rel_change={max_rel:.3e}")

        # Apply swaps: update network for next round
        for id_firm in range(n):
            if aa_do[id_firm]:
                removed = aa_remove[id_firm]
                added = aa_add[id_firm]
                # Update supplier list
                supplier_id_list[id_firm].remove(removed)
                supplier_id_list[id_firm].append(added)
                supplier_id_list[id_firm].sort()
                # Update alternate list: remove added, add removed
                alternate_supplier_id_list[id_firm].remove(added)
                alternate_supplier_id_list[id_firm].append(removed)

        # Rebuild W from updated supplier lists
        W = build_W_from_suppliers(supplier_id_list, Wbar)

        if max_rel < CONVERGENCE_THRESHOLD:
            print(f"\n*** Converged after {r} rounds ***")
            converged = True
            final_round = r
            break

        P_ref = P_next

    # Final equilibrium
    W_final = build_W_from_suppliers(supplier_id_list, Wbar)
    final_z = compute_adjusted_z(AiSi, supplier_id_list)
    final_eq = compute_equilibrium(a, final_z, W_final, n)
    final_utility = -final_eq['P'].sum()

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Converged: {converged}, Rounds: {final_round}, Time: {(datetime.now() - starting_time).total_seconds():.2f}s")
    print(f"Utility: {initial_utility:.4f} -> {final_utility:.4f} ({(final_utility - initial_utility) / abs(initial_utility) * 100:.2f}%)")
    print(f"Prices AA, min max: [{aa_best_cost.min():.4f}, {aa_best_cost.max():.4f}]")
    print(f"Prices GE, min max: [{final_eq['P'].min():.4f}, {final_eq['P'].max():.4f}]")
    print(f"Max |P_GE - P_AA| / P_AA: {np.max(np.abs(final_eq['P'] - aa_best_cost) / aa_best_cost):.3e}")

    return {
        'converged': converged, 'rounds': final_round,
        'initial_utility': initial_utility, 'final_utility': final_utility,
        'final_prices': final_eq['P'], 'aa_best_cost': aa_best_cost,
        'final_supplier_list': supplier_id_list,
    }


if __name__ == "__main__":
    run_aa_simulation()
