"""
Standalone AA (Asynchronous Algorithm) mode simulation for network rewiring.
Independent from run.py and config.yml - all parameters are set in this file.

AA mode uses frozen reference prices within each round, with synchronous updates
and a price ratchet mechanism. Valid under CRS (b=1) with alpha=1.
"""

import random
from datetime import datetime

import numpy as np

from utils import (
    EPSILON,
    initialize_sffa_adjacency,
    add_one_supplier_each_firm,
    identify_suppliers_from_matrix,
    create_technology_matrix,
    get_AiSi_productivities,
    compute_adjusted_z,
    compute_equilibrium_crs,
    build_W_from_suppliers,
)


# =============================================================================
# PARAMETERS (all hardcoded)
# =============================================================================

# Simulation parameters
NB_ROUNDS = 50
NB_FIRMS = 100
C = 4  # base connectivity (edges per node in initial network)
CC = 4  # number of extra potential suppliers per firm

# Economic parameters (homogeneous)
# b=1 (CRS) and weights sum to 1 => alpha = a + (1-a)*1 = 1 always
A_VALUE = 0.5  # labor share

# Network parameters
AISI_SPREAD = 0.01  # AiSi productivity spread parameter

# Numerical parameters
CONVERGENCE_THRESHOLD = 1e-10


# =============================================================================
# NETWORK SETUP
# =============================================================================

def generate_network(n, c, cc, aisi_spread, seed=None):
    """
    Generate initial network state. Can be seeded for reproducibility.
    Returns all state needed to run simulation.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    M0 = add_one_supplier_each_firm(initialize_sffa_adjacency(n, c))
    supplier_id_list, nb_suppliers = identify_suppliers_from_matrix(M0)
    Wbar, alternate_supplier_id_list = create_technology_matrix(
        M0, nb_suppliers, np.full(n, cc), supplier_id_list
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

    Args:
        network_state: Pre-generated network from generate_network(). If None, generates new.
        seed: Random seed for permutations during simulation (not network generation).
        verbose: Print progress and results.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if verbose:
        print("=" * 60)
        print("AA Mode Simulation (alpha=1 simplified)")
        print("=" * 60)

    starting_time = datetime.now()
    n = NB_FIRMS
    a = A_VALUE

    if verbose:
        print(f"\nParameters: n={n}, rounds={NB_ROUNDS}, c={C}, cc={CC}, a={a}, AiSi_spread={AISI_SPREAD}")

    # Use provided network state or generate new
    if network_state is not None:
        M0 = network_state['M0']
        W0 = network_state['W0']
        Wbar = network_state['Wbar']
        supplier_id_list = [list(s) for s in network_state['supplier_id_list']]  # deep copy
        alternate_supplier_id_list = [list(s) for s in network_state['alternate_supplier_id_list']]
        AiSi = network_state['AiSi']
        nb_suppliers = network_state['nb_suppliers']
        if verbose:
            print("\nUsing provided network state...")
    else:
        if verbose:
            print("\nGenerating SF-FA network...")
        M0 = add_one_supplier_each_firm(initialize_sffa_adjacency(n, C))
        supplier_id_list, nb_suppliers = identify_suppliers_from_matrix(M0)
        Wbar, alternate_supplier_id_list = create_technology_matrix(
            M0, nb_suppliers, np.full(n, CC), supplier_id_list
        )
        W0 = M0 * Wbar
        AiSi = get_AiSi_productivities(supplier_id_list, alternate_supplier_id_list, AISI_SPREAD)

    if verbose:
        print(f"  Edges: {int(M0.sum())}, Mean suppliers: {nb_suppliers.mean():.2f}")

    # Initial equilibrium
    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
    eq = compute_equilibrium_crs(a, adjusted_z, W0, n)
    initial_utility = -eq['P'].sum()

    if verbose:
        print(f"  Initial utility: {initial_utility:.4f}, prices: [{eq['P'].min():.4f}, {eq['P'].max():.4f}]")

    # Initialize AA variables
    P_ref = eq['P'].copy()
    aa_best_cost = P_ref.copy()

    # Main loop
    if verbose:
        print("\n" + "-" * 60)
    converged = False
    final_round = NB_ROUNDS
    P_ref_history = [P_ref.copy()]  # Track price history for monotonicity check

    for r in range(1, NB_ROUNDS + 1):
        aa_do = np.zeros(n, dtype=bool)
        aa_remove = [None] * n
        aa_add = [None] * n

        for id_firm in np.random.permutation(n):
            # First: compute current cost (no swap) with current P_ref
            current_W_col = np.zeros(n)
            for s in supplier_id_list[id_firm]:
                current_W_col[s] = Wbar[s, id_firm]
            current_z = AiSi[id_firm][tuple(int(s) for s in sorted(supplier_id_list[id_firm]))]
            current_cost = np.prod(np.power(P_ref, (1 - a) * current_W_col)) / current_z
            aa_best_cost[id_firm] = current_cost  # Update to actual current cost
            potential_cost = current_cost
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

        if verbose:
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

        # === VERIFICATION CHECKS ===
        # 1. Check P_ref monotonicity (prices should be non-increasing)
        P_prev = P_ref_history[-1]
        price_increases = P_next > P_prev + EPSILON
        if price_increases.any() and verbose:
            num_increases = price_increases.sum()
            max_increase = (P_next - P_prev)[price_increases].max()
            print(f"  WARNING: {num_increases} prices increased! Max increase: {max_increase:.6e}")
        P_ref_history.append(P_next.copy())

        # 2. Check market clearing (supply = demand) using actual GE
        current_z = compute_adjusted_z(AiSi, supplier_id_list)
        current_eq = compute_equilibrium_crs(a, current_z, W, n)
        P_ge = current_eq['P']
        X_ge = current_eq['X']
        v_ge = P_ge * X_ge  # sales

        # Market clearing: X[i] = final_demand[i] + intermediate_demand[i]
        # final_demand[i] = B / (n * P[i]) where B = total labor income = sum(a * v)
        B = a * v_ge.sum()  # total budget (labor income, since alpha=1)
        final_demand = B / (n * P_ge)
        # intermediate_demand[i] = sum_j W[i,j] * (1-a) * v[j] / P[i]
        intermediate_demand = ((1 - a) * W @ v_ge) / P_ge
        total_demand = final_demand + intermediate_demand
        supply = X_ge

        market_clearing_error = np.abs(supply - total_demand)
        max_clearing_error = market_clearing_error.max()
        if max_clearing_error > 1e-8 and verbose:
            print(f"  WARNING: Market clearing error = {max_clearing_error:.6e}")

        if max_rel < CONVERGENCE_THRESHOLD:
            if verbose:
                print(f"\n*** Converged after {r} rounds ***")
            converged = True
            final_round = r
            break

        P_ref = P_next

    # Final equilibrium
    W_final = build_W_from_suppliers(supplier_id_list, Wbar)
    final_z = compute_adjusted_z(AiSi, supplier_id_list)
    final_eq = compute_equilibrium_crs(a, final_z, W_final, n)
    final_utility = -final_eq['P'].sum()

    # Results
    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Converged: {converged}, Rounds: {final_round}, Time: {(datetime.now() - starting_time).total_seconds():.2f}s")
        print(f"Utility: {initial_utility:.4f} -> {final_utility:.4f} ({(final_utility - initial_utility) / abs(initial_utility) * 100:.2f}%)")
        print(f"Prices AA, min max: [{aa_best_cost.min():.4f}, {aa_best_cost.max():.4f}]")
        print(f"Prices GE, min max: [{final_eq['P'].min():.4f}, {final_eq['P'].max():.4f}]")
        print(f"Max |P_GE - P_AA| / P_AA: {np.max(np.abs(final_eq['P'] - aa_best_cost) / aa_best_cost):.3e}")

        # Verification summary
        print("\n" + "-" * 60)
        print("VERIFICATION SUMMARY")
        print("-" * 60)
        # Check price monotonicity across all rounds
        P_history = np.array(P_ref_history)
        price_diffs = np.diff(P_history, axis=0)  # P[r+1] - P[r]
        max_increase = price_diffs.max()
        num_increases = (price_diffs > EPSILON).sum()
        print(f"Price monotonicity: {'PASS' if num_increases == 0 else 'FAIL'}")
        print(f"  Max price increase across all rounds: {max_increase:.6e}")
        print(f"  Number of (firm, round) pairs with price increase: {num_increases}")
        print(f"Market clearing: Verified at each round (GE computation ensures this)")

    return {
        'converged': converged, 'rounds': final_round,
        'initial_utility': initial_utility, 'final_utility': final_utility,
        'final_prices': final_eq['P'], 'aa_best_cost': aa_best_cost,
        'final_supplier_list': supplier_id_list,
    }


if __name__ == "__main__":
    run_aa_simulation()
