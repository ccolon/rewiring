"""
Standalone AA (Asynchronous Algorithm) mode simulation for network rewiring.
Independent from run.py and config.yml - all parameters are set in this file.

AA mode uses frozen reference prices within each round, with synchronous updates.
Firms evaluate swaps using: cost = prod(P_ref^((1-a)*W)) / z_i

Supports both CRS (b=1) and non-CRS (b!=1) cases:
- CRS: uses compute_equilibrium_crs, prices are naive cost estimates
- Non-CRS: uses compute_equilibrium_full, true equilibrium recomputed each round
"""

import random
from datetime import datetime
from itertools import combinations

import numpy as np

from utils import (
    EPSILON,
    generate_parameter,
    generate_a_parameter,
    initialize_sffa_adjacency,
    add_one_supplier_each_firm,
    identify_suppliers_from_matrix,
    create_technology_matrix,
    get_AiSi_productivities,
    compute_adjusted_z,
    compute_equilibrium_crs,
    compute_equilibrium_full,
    build_W_from_suppliers,
    calculate_utility,
)


# =============================================================================
# PARAMETERS (all hardcoded)
# =============================================================================

# Simulation parameters
NB_ROUNDS = 20
NB_FIRMS = 20
C = 4  # base connectivity (edges per node in initial network)
CC = 1  # number of extra potential suppliers per firm
MAX_SWAPS = 1  # max simultaneous supplier swaps (1 = single, 2 = up to dual)

# Economic parameters
B_CONFIG = {'mode': 'homogeneous', 'value': 0.9}  # returns to scale (None or 1.0 for CRS)
A_CONFIG = {'mode': 'homogeneous', 'value': 0.5}  # labor share
Z_CONFIG = {'mode': 'homogeneous', 'value': 1.0}  # base productivity
SIGMA_W = 0.0  # weight variance

# Network parameters
AISI_SPREAD = 0.1  # AiSi productivity spread parameter

# Numerical parameters
CONVERGENCE_THRESHOLD = 1e-10


# =============================================================================
# NETWORK SETUP
# =============================================================================

def generate_network(n, c, cc, aisi_spread, seed=None, a=None, b=None, sigma_w=0.0):
    """
    Generate initial network state. Can be seeded for reproducibility.
    Returns all state needed to run simulation.

    Args:
        n: Number of firms
        c: Base connectivity
        cc: Number of alternate suppliers per firm
        aisi_spread: AiSi productivity spread
        seed: Random seed
        a: Labor share array (for weight bounds in non-CRS mode)
        b: Returns to scale array (for weight bounds in non-CRS mode)
        sigma_w: Weight variance
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    M0 = add_one_supplier_each_firm(initialize_sffa_adjacency(n, c))
    supplier_id_list, nb_suppliers = identify_suppliers_from_matrix(M0)
    Wbar, alternate_supplier_id_list = create_technology_matrix(
        M0, nb_suppliers, np.full(n, cc), supplier_id_list,
        a=a, b=b, sigma_w=sigma_w
    )
    W0 = M0 * Wbar
    AiSi = get_AiSi_productivities(supplier_id_list, alternate_supplier_id_list, aisi_spread)

    return {
        'M0': M0,
        'W0': W0,
        'Wbar': Wbar,
        'supplier_id_list': [list(s) for s in supplier_id_list],  # deep copy
        'alternate_supplier_id_list': [list(s) for s in alternate_supplier_id_list],
        'AiSi': AiSi,
        'nb_suppliers': nb_suppliers,
    }


# =============================================================================
# MAIN AA SIMULATION
# =============================================================================

def run_aa_simulation(network_state=None, seed=None, verbose=True):
    """
    Run the AA mode simulation.

    Firms evaluate swaps using frozen reference prices (AA principle):
        cost = prod(P_ref^((1-a)*W_col)) / z_i

    Supports CRS (B_CONFIG value=1.0) and non-CRS (e.g. B_CONFIG value=0.9):
    - CRS: equilibrium via compute_equilibrium_crs, naive cost prices
    - Non-CRS: true equilibrium recomputed via compute_equilibrium_full each round

    Args:
        network_state: Pre-generated network from generate_network(). If None, generates new.
        seed: Random seed for permutations during simulation (not network generation).
        verbose: Print progress and results.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    starting_time = datetime.now()
    n = NB_FIRMS
    max_swaps = MAX_SWAPS

    # Generate economic parameters
    b = generate_parameter(B_CONFIG, n, 'b', verbose=verbose)
    a = generate_a_parameter(A_CONFIG, b, n, verbose=verbose)
    z = generate_parameter(Z_CONFIG, n, 'z', verbose=verbose)

    # Determine if CRS mode (all b values are 1.0)
    crs_mode = np.allclose(b, 1.0)

    if verbose:
        print("=" * 60)
        print(f"AA Mode Simulation ({'CRS' if crs_mode else f'non-CRS, b={B_CONFIG}'})")
        print("=" * 60)
        print(f"\nParameters: n={n}, rounds={NB_ROUNDS}, c={C}, cc={CC}, "
              f"max_swaps={max_swaps}, AiSi_spread={AISI_SPREAD}")
        print(f"  a*b: max={np.max(a * b):.4f} (must be < 1)")

    # Use provided network state or generate new
    if network_state is not None:
        M0 = network_state['M0']
        W0 = network_state['W0']
        Wbar = network_state['Wbar']
        supplier_id_list = [list(s) for s in network_state['supplier_id_list']]
        alternate_supplier_id_list = [list(s) for s in network_state['alternate_supplier_id_list']]
        AiSi = network_state['AiSi']
        nb_suppliers = network_state['nb_suppliers']
        if verbose:
            print("\nUsing provided network state...")
    else:
        if verbose:
            print("\nGenerating SF-FA network...")
        state = generate_network(n, C, CC, AISI_SPREAD, a=a, b=b, sigma_w=SIGMA_W)
        M0 = state['M0']
        W0 = state['W0']
        Wbar = state['Wbar']
        supplier_id_list = [list(s) for s in state['supplier_id_list']]
        alternate_supplier_id_list = [list(s) for s in state['alternate_supplier_id_list']]
        AiSi = state['AiSi']
        nb_suppliers = state['nb_suppliers']

    if verbose:
        print(f"  Edges: {int(M0.sum())}, Mean suppliers: {nb_suppliers.mean():.2f}")

    # Initial equilibrium
    if crs_mode:
        adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
        eq = compute_equilibrium_crs(a[0] if isinstance(a, np.ndarray) else a,
                                     adjusted_z, W0, n)
        initial_utility = -eq['P'].sum()
    else:
        adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
        eq = compute_equilibrium_full(a, b, adjusted_z, W0, n)
        initial_utility = calculate_utility(eq)

    if verbose:
        print(f"  Initial utility: {initial_utility:.4f}, prices: [{eq['P'].min():.4f}, {eq['P'].max():.4f}]")

    # Initialize AA variables
    P_ref = eq['P'].copy()
    one_minus_a = (1 - a)  # array

    # Main loop
    if verbose:
        print("\n" + "-" * 60)
    converged = False
    final_round = NB_ROUNDS
    P_ref_history = [P_ref.copy()]

    for r in range(1, NB_ROUNDS + 1):
        aa_do = np.zeros(n, dtype=bool)
        aa_removes = [None] * n
        aa_adds = [None] * n

        for id_firm in np.random.permutation(n):
            oma = one_minus_a[id_firm]

            # Compute current cost with frozen P_ref
            current_W_col = np.zeros(n)
            for s in supplier_id_list[id_firm]:
                current_W_col[s] = Wbar[s, id_firm]
            current_z = AiSi[id_firm][tuple(int(s) for s in sorted(supplier_id_list[id_firm]))]
            current_cost = np.prod(np.power(P_ref, oma * current_W_col)) / current_z
            potential_cost = current_cost
            best_removes, best_adds = None, None

            current_set = set(supplier_id_list[id_firm])
            alternates = alternate_supplier_id_list[id_firm]

            # Evaluate all swap combinations
            for swap_size in range(1, max_swaps + 1):
                if len(alternates) < swap_size or len(supplier_id_list[id_firm]) < swap_size:
                    continue
                for new_sups in combinations(alternates, swap_size):
                    for old_sups in combinations(supplier_id_list[id_firm], swap_size):
                        new_set = (current_set - set(old_sups)) | set(new_sups)
                        W_col = np.zeros(n)
                        for s in new_set:
                            W_col[s] = Wbar[s, id_firm]
                        test_z = AiSi[id_firm][tuple(sorted(int(s) for s in new_set))]
                        cost = np.prod(np.power(P_ref, oma * W_col)) / test_z

                        if cost < potential_cost - EPSILON:
                            potential_cost = cost
                            best_removes, best_adds = list(old_sups), list(new_sups)

            if best_adds is not None:
                aa_do[id_firm] = True
                aa_removes[id_firm] = best_removes
                aa_adds[id_firm] = best_adds

        # Count rewirings
        rewirings_this_round = aa_do.sum()

        if verbose:
            print(f"Round {r:3d}: swaps={rewirings_this_round:3d}", end="")

        # Apply swaps
        for id_firm in range(n):
            if aa_do[id_firm]:
                for old_s, new_s in zip(aa_removes[id_firm], aa_adds[id_firm]):
                    supplier_id_list[id_firm].remove(old_s)
                    supplier_id_list[id_firm].append(new_s)
                    alternate_supplier_id_list[id_firm].remove(new_s)
                    alternate_supplier_id_list[id_firm].append(old_s)
                supplier_id_list[id_firm].sort()

        # Rebuild W
        W = build_W_from_suppliers(supplier_id_list, Wbar)

        # Convergence: no rewiring happened
        if rewirings_this_round == 0:
            if verbose:
                print(f"\n\n*** Converged after {r} rounds ***")
            converged = True
            final_round = r
            break

        # Recompute true equilibrium
        if crs_mode:
            current_z = compute_adjusted_z(AiSi, supplier_id_list)
            eq = compute_equilibrium_crs(a[0] if isinstance(a, np.ndarray) else a,
                                         current_z, W, n)
        else:
            adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
            eq = compute_equilibrium_full(a, b, adjusted_z, W, n)

        P_next = eq['P'].copy()

        # Track price changes
        max_rel = np.max(np.abs(P_next - P_ref) / np.maximum(P_ref, 1e-12))
        if verbose:
            print(f", max_rel_change={max_rel:.3e}")

        # Check price monotonicity
        P_prev = P_ref_history[-1]
        price_increases = P_next > P_prev + EPSILON
        if price_increases.any() and verbose:
            num_increases = price_increases.sum()
            max_increase = (P_next - P_prev)[price_increases].max()
            print(f"  WARNING: {num_increases} prices increased! Max increase: {max_increase:.6e}")
        P_ref_history.append(P_next.copy())

        P_ref = P_next

    # Final equilibrium
    W_final = build_W_from_suppliers(supplier_id_list, Wbar)
    if crs_mode:
        final_z = compute_adjusted_z(AiSi, supplier_id_list)
        final_eq = compute_equilibrium_crs(a[0] if isinstance(a, np.ndarray) else a,
                                           final_z, W_final, n)
        final_utility = -final_eq['P'].sum()
    else:
        final_z = compute_adjusted_z(AiSi, supplier_id_list, z)
        final_eq = compute_equilibrium_full(a, b, final_z, W_final, n)
        final_utility = calculate_utility(final_eq)

    # Results
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Converged: {converged}, Rounds: {final_round}, Time: {(datetime.now() - starting_time).total_seconds():.2f}s")
        print(f"Utility: {initial_utility:.4f} -> {final_utility:.4f} ({(final_utility - initial_utility) / abs(initial_utility) * 100:.2f}%)")
        print(f"Prices GE, min max: [{final_eq['P'].min():.4f}, {final_eq['P'].max():.4f}]")

        # Verification summary
        print("\n" + "-" * 60)
        print("VERIFICATION SUMMARY")
        print("-" * 60)
        P_history = np.array(P_ref_history)
        price_diffs = np.diff(P_history, axis=0)
        max_increase = price_diffs.max()
        num_increases = (price_diffs > EPSILON).sum()
        print(f"Price monotonicity: {'PASS' if num_increases == 0 else 'FAIL'}")
        print(f"  Max price increase across all rounds: {max_increase:.6e}")
        print(f"  Number of (firm, round) pairs with price increase: {num_increases}")

    return {
        'converged': converged, 'rounds': final_round,
        'initial_utility': initial_utility, 'final_utility': final_utility,
        'final_prices': final_eq['P'],
        'final_supplier_list': supplier_id_list,
    }


if __name__ == "__main__":
    run_aa_simulation()
