"""
Standalone Full Anticipation mode simulation for network rewiring.
Independent from run.py and config.yml - all parameters are set in this file.

Full anticipation: firms compute full network-wide equilibrium when evaluating
each potential supplier switch. Swaps are applied immediately (sequential updates).

Supports heterogeneous parameters (a, b, z) with modes: homogeneous, uniform, normal.
"""

import copy
import random
from datetime import datetime

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
    compute_equilibrium_full,
    build_W_from_suppliers,
    calculate_utility,
)


# =============================================================================
# PARAMETERS (all hardcoded - modify here)
# =============================================================================

# Simulation parameters
NB_ROUNDS = 50
NB_FIRMS = 50
C = 4  # base connectivity (edges per node in initial network)
CC = 1  # number of extra potential suppliers per firm

# Economic parameters configuration
# Each parameter can be: homogeneous, uniform, or normal
# Format: {'mode': 'homogeneous'|'uniform'|'normal', ...}

B_CONFIG = {
    'mode': 'homogeneous',  # 'homogeneous', 'uniform', or 'normal'
    'value': 1.0,           # for homogeneous
    'min': 0.8,             # for uniform
    'max': 0.95,            # for uniform
    'mean': 0.9,            # for normal
    'sigma': 0.05,          # for normal
    'bound_min': 0.5,       # for normal (clipping)
    'bound_max': 1.0,       # for normal (clipping)
}

A_CONFIG = {
    'mode': 'homogeneous',
    'value': 0.5,
    'min': 0.3,
    'max': 0.7,
    'mean': 0.5,
    'sigma': 0.1,
    'bound_min': 0.05,
    'bound_max': 0.95,
}

Z_CONFIG = {
    'mode': 'homogeneous',
    'value': 1.0,
    'min': 0.5,
    'max': 1.5,
    'mean': 1.0,
    'sigma': 0.1,
    'bound_min': 0.1,
    'bound_max': 2.0,
}

# Network parameters
SIGMA_W = 0.0  # Weight variance (0 = uniform weights)
AISI_SPREAD = 0.1  # AiSi productivity spread parameter


# =============================================================================
# MAIN FULL ANTICIPATION SIMULATION
# =============================================================================

def run_full_simulation(seed=None, verbose=True):
    """Run the full anticipation mode simulation."""

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if verbose:
        print("=" * 60)
        print("Full Anticipation Mode Simulation")
        print("=" * 60)

    starting_time = datetime.now()
    n = NB_FIRMS

    if verbose:
        print(f"\nParameters: n={n}, rounds={NB_ROUNDS}, c={C}, cc={CC}")
        print(f"AiSi_spread={AISI_SPREAD}, sigma_w={SIGMA_W}")

    # Generate economic parameters
    if verbose:
        print("\nGenerating economic parameters...")
    b = generate_parameter(B_CONFIG, n, 'b', verbose=verbose)
    a = generate_a_parameter(A_CONFIG, b, n, verbose=verbose)
    z = generate_parameter(Z_CONFIG, n, 'z', verbose=verbose)

    # Verify constraints
    if verbose:
        print(f"\n  a*b: max={np.max(a * b):.4f} (must be < 1)")
        print(f"  (1-a)*b: max={np.max((1-a) * b):.4f}")

    # Generate network
    if verbose:
        print("\nGenerating SF-FA network...")
    M0 = add_one_supplier_each_firm(initialize_sffa_adjacency(n, C))
    supplier_id_list, nb_suppliers = identify_suppliers_from_matrix(M0)
    Wbar, alternate_supplier_id_list = create_technology_matrix(
        M0, nb_suppliers, np.full(n, CC), supplier_id_list,
        a=a, b=b, sigma_w=SIGMA_W
    )
    W0 = M0 * Wbar
    AiSi = get_AiSi_productivities(supplier_id_list, alternate_supplier_id_list, AISI_SPREAD)

    if verbose:
        print(f"  Edges: {int(M0.sum())}, Mean suppliers: {nb_suppliers.mean():.2f}")

    # Initial equilibrium
    W = W0.copy()
    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
    eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
    initial_utility = calculate_utility(eq)

    if verbose:
        print(f"  Initial utility: {initial_utility:.4f}")
        print(f"  Initial prices: [{eq['P'].min():.4f}, {eq['P'].max():.4f}], {eq['P']}")

    # Main simulation loop
    if verbose:
        print("\n" + "-" * 60)
        print("Starting simulation rounds...")
        print("-" * 60)

    total_rewirings = 0
    converged = False
    final_round = NB_ROUNDS
    price_history = [eq['P'].copy()]  # Track price history for monotonicity check

    for r in range(1, NB_ROUNDS + 1):
        rewirings_this_round = 0
        rewiring_order = np.random.permutation(n)

        for id_firm in rewiring_order:
            current_cost = eq['P'][id_firm]
            potential_cost = current_cost
            best_remove, best_add = None, None

            # Evaluate all possible swaps
            for new_sup in alternate_supplier_id_list[id_firm]:
                # Temporarily add new supplier
                W[new_sup, id_firm] = Wbar[new_sup, id_firm]

                for old_sup in supplier_id_list[id_firm]:
                    # Temporarily remove old supplier
                    W[old_sup, id_firm] = 0

                    # Compute new equilibrium with this swap
                    tmp_supplier_list = copy.deepcopy(supplier_id_list)
                    tmp_supplier_list[id_firm] = sorted(
                        (set(supplier_id_list[id_firm]) | {new_sup}) - {old_sup}
                    )
                    tmp_adjusted_z = compute_adjusted_z(AiSi, tmp_supplier_list, z)
                    new_eq = compute_equilibrium_full(a, b, tmp_adjusted_z, W, n)
                    estimated_cost = new_eq['P'][id_firm]

                    if estimated_cost < potential_cost - EPSILON:
                        potential_cost = estimated_cost
                        best_remove, best_add = old_sup, new_sup

                    # Restore old supplier
                    W[old_sup, id_firm] = Wbar[old_sup, id_firm]

                # Remove new supplier
                W[new_sup, id_firm] = 0

            # Apply best swap if found
            if best_add is not None:
                # Update W
                W[best_add, id_firm] = Wbar[best_add, id_firm]
                W[best_remove, id_firm] = 0

                # Update supplier lists
                supplier_id_list[id_firm].remove(best_remove)
                supplier_id_list[id_firm].append(best_add)
                alternate_supplier_id_list[id_firm].remove(best_add)
                alternate_supplier_id_list[id_firm].append(best_remove)

                # Recompute equilibrium
                adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
                eq = compute_equilibrium_full(a, b, adjusted_z, W, n)

                rewirings_this_round += 1
                total_rewirings += 1

                if verbose:
                    print(f"  Firm {id_firm}: {best_remove}->{best_add}, "
                          f"cost {current_cost:.4f}->{eq['P'][id_firm]:.4f}")

        # Check price monotonicity (prices should be non-increasing)
        P_prev = price_history[-1]
        P_curr = eq['P']
        price_increases = P_curr > P_prev + EPSILON
        if price_increases.any() and verbose:
            num_increases = price_increases.sum()
            max_increase = (P_curr - P_prev)[price_increases].max()
            print(f"  WARNING: {num_increases} prices increased! Max increase: {max_increase:.6e}")
        price_history.append(P_curr.copy())

        if verbose:
            print(f"Round {r:3d}: {rewirings_this_round} swaps, "
                  f"utility={calculate_utility(eq):.4f}, "
                  f"prices=[{eq['P'].min():.4f}, {eq['P'].max():.4f}]")

        # Check convergence
        if rewirings_this_round == 0:
            if verbose:
                print(f"\n*** Converged after {r} rounds ***")
            converged = True
            final_round = r
            break

    # Final results
    final_utility = calculate_utility(eq)
    total_time = (datetime.now() - starting_time).total_seconds()

    if verbose:
        print("\n" + "=" * 60)
        print("RESULTS")
        print("=" * 60)
        print(f"Converged: {converged}, Rounds: {final_round}, Total rewirings: {total_rewirings}")
        print(f"Time: {total_time:.2f}s")
        print(f"Utility: {initial_utility:.4f} -> {final_utility:.4f} "
              f"({(final_utility - initial_utility) / abs(initial_utility) * 100:.2f}%)")
        print(f"Final prices: [{eq['P'].min():.4f}, {eq['P'].max():.4f}]")

        # Verification summary
        print("\n" + "-" * 60)
        print("VERIFICATION SUMMARY")
        print("-" * 60)
        # Check price monotonicity across all rounds
        P_history_arr = np.array(price_history)
        price_diffs = np.diff(P_history_arr, axis=0)  # P[r+1] - P[r]
        max_increase = price_diffs.max()
        num_increases = (price_diffs > EPSILON).sum()
        print(f"Price monotonicity: {'PASS' if num_increases == 0 else 'FAIL'}")
        print(f"  Max price increase across all rounds: {max_increase:.6e}")
        print(f"  Number of (firm, round) pairs with price increase: {num_increases}")

    return {
        'converged': converged,
        'rounds': final_round,
        'total_rewirings': total_rewirings,
        'time': total_time,
        'initial_utility': initial_utility,
        'final_utility': final_utility,
        'final_prices': eq['P'],
        'final_supplier_list': supplier_id_list,
        'a': a, 'b': b, 'z': z,
    }


if __name__ == "__main__":
    result = run_full_simulation()
