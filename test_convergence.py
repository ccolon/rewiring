"""
Test script to verify convergence properties for AA and Full anticipation modes.

Two test modes:
1. same_start_different_order:
   Same initial network, different permutation orders
   -> Tests that evaluation order doesn't affect result

2. different_start_same_parameter:
   Different initial networks, same Wbar/AiSi
   -> Tests whether equilibrium is unique (path-independent)

Two simulation modes:
- aa: AA (Asynchronous Algorithm) mode with frozen reference prices
- full: Full anticipation mode with full equilibrium recomputation

Compare mode (--compare):
- Runs both AA and Full on the same network(s)
- Compares whether they reach the same equilibrium
- Uses consistent parameters: a=0.5, b=1.0, z=1.0
"""

import copy
import random
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
# TEST CONFIGURATION (modify here to run without command line)
# =============================================================================

SIM_MODE = "full"  # "aa" or "full"
TEST_MODE = "different_start_same_parameter"  # "same_start_different_order" or "different_start_same_parameter"
COMPARE_MODES = False  # If True, compare AA vs Full instead of running single mode

# Shared parameters
NB_FIRMS = 20
NB_ROUNDS = 20
C = 4
CC = 2
AISI_SPREAD = 0.1
MAX_SWAPS = 2  # max number of simultaneous supplier swaps (1 = single, 2 = up to dual)

# AA mode parameters
A_VALUE = 0.5  # homogeneous labor share for AA mode

# Full mode parameters (can be homogeneous or heterogeneous)
B_CONFIG = {'mode': 'homogeneous', 'value': 1.0}
A_CONFIG = {'mode': 'homogeneous', 'value': 0.5}
Z_CONFIG = {'mode': 'homogeneous', 'value': 1.0}
SIGMA_W = 0.0

# b-sweep: homogeneous b values to test for convergence
B_VALUES = [0.9, 1.0, 1.1]

# Test parameters
N_TRIALS = 10
NETWORK_SEED = 123
SHOW_COMPARISON = False

# Numerical
CONVERGENCE_THRESHOLD = 1e-10


# =============================================================================
# NETWORK GENERATION
# =============================================================================

def generate_base_network(n, c, cc, aisi_spread, seed=None, a=None, b=None, sigma_w=0.0):
    """
    Generate initial network state. Can be seeded for reproducibility.

    Args:
        n: Number of firms
        c: Base connectivity
        cc: Number of alternate suppliers per firm
        aisi_spread: AiSi productivity spread
        seed: Random seed
        a: Labor share array (for full mode weight bounds)
        b: Returns to scale array (for full mode weight bounds)
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
        'supplier_id_list': [list(s) for s in supplier_id_list],
        'alternate_supplier_id_list': [list(s) for s in alternate_supplier_id_list],
        'AiSi': AiSi,
        'nb_suppliers': nb_suppliers,
    }


def generate_random_initial_network(n, Wbar, AiSi, seed):
    """
    Generate a random initial network by selecting random supplier combinations
    from the valid combinations in AiSi.
    """
    random.seed(seed)
    np.random.seed(seed)

    new_supplier_list = []
    new_alternate_list = []

    for firm in range(n):
        all_combinations = list(AiSi[firm].keys())
        chosen_combo = list(random.choice(all_combinations))
        new_supplier_list.append(chosen_combo)

        all_potential = set()
        for combo in all_combinations:
            all_potential.update(combo)
        new_alternate_list.append(list(all_potential - set(chosen_combo)))

    M0 = np.zeros((n, n))
    for buyer, suppliers in enumerate(new_supplier_list):
        for supplier in suppliers:
            M0[supplier, buyer] = 1

    W0 = M0 * Wbar

    return {
        'M0': M0,
        'W0': W0,
        'Wbar': Wbar,
        'supplier_id_list': new_supplier_list,
        'alternate_supplier_id_list': new_alternate_list,
        'AiSi': AiSi,
        'nb_suppliers': np.array([len(s) for s in new_supplier_list]),
    }


# =============================================================================
# AA MODE SIMULATION
# =============================================================================

def run_aa_simulation(network_state, a, seed=None, max_swaps=1, nb_rounds=None):
    """Run AA mode simulation (simplified, for testing).

    Args:
        max_swaps: Maximum number of simultaneous supplier swaps (1=single, 2=up to dual).
        nb_rounds: Number of rounds (defaults to module-level NB_ROUNDS).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = len(network_state['AiSi'])
    Wbar = network_state['Wbar']
    AiSi = network_state['AiSi']
    supplier_id_list = [list(s) for s in network_state['supplier_id_list']]
    alternate_supplier_id_list = [list(s) for s in network_state['alternate_supplier_id_list']]
    W = network_state['W0'].copy()

    # Initial equilibrium
    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
    eq = compute_equilibrium_crs(a, adjusted_z, W, n)
    initial_utility = -eq['P'].sum()

    P_ref = eq['P'].copy()
    aa_best_cost = P_ref.copy()

    _nb_rounds = nb_rounds if nb_rounds is not None else NB_ROUNDS
    for r in range(1, _nb_rounds + 1):
        aa_do = np.zeros(n, dtype=bool)
        aa_removes = [None] * n  # list of suppliers to remove (length 1 or 2)
        aa_adds = [None] * n     # list of suppliers to add (length 1 or 2)

        for id_firm in np.random.permutation(n):
            current_W_col = np.zeros(n)
            for s in supplier_id_list[id_firm]:
                current_W_col[s] = Wbar[s, id_firm]
            current_z = AiSi[id_firm][tuple(int(s) for s in sorted(supplier_id_list[id_firm]))]
            current_cost = np.prod(np.power(P_ref, (1 - a) * current_W_col)) / current_z
            aa_best_cost[id_firm] = current_cost
            potential_cost = current_cost
            best_removes, best_adds = None, None

            current_set = set(supplier_id_list[id_firm])
            alternates = alternate_supplier_id_list[id_firm]

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
                        cost = np.prod(np.power(P_ref, (1 - a) * W_col)) / test_z

                        if cost < potential_cost - EPSILON:
                            potential_cost = cost
                            best_removes, best_adds = list(old_sups), list(new_sups)

            if best_adds is not None:
                aa_do[id_firm] = True
                aa_removes[id_firm] = best_removes
                aa_adds[id_firm] = best_adds
                aa_best_cost[id_firm] = potential_cost

        P_next = aa_best_cost.copy()
        max_rel = np.max(np.abs(P_next - P_ref) / np.maximum(P_ref, 1e-12))

        for id_firm in range(n):
            if aa_do[id_firm]:
                for old_s, new_s in zip(aa_removes[id_firm], aa_adds[id_firm]):
                    supplier_id_list[id_firm].remove(old_s)
                    supplier_id_list[id_firm].append(new_s)
                    alternate_supplier_id_list[id_firm].remove(new_s)
                    alternate_supplier_id_list[id_firm].append(old_s)
                supplier_id_list[id_firm].sort()

        W = build_W_from_suppliers(supplier_id_list, Wbar)

        if max_rel < CONVERGENCE_THRESHOLD:
            break

        P_ref = P_next

    # Final equilibrium
    final_z = compute_adjusted_z(AiSi, supplier_id_list)
    final_eq = compute_equilibrium_crs(a, final_z, W, n)

    return {
        'converged': max_rel < CONVERGENCE_THRESHOLD,
        'rounds': r,
        'initial_utility': initial_utility,
        'final_utility': -final_eq['P'].sum(),
        'final_prices': final_eq['P'],
        'final_supplier_list': supplier_id_list,
    }


# =============================================================================
# FULL MODE SIMULATION
# =============================================================================

def run_full_simulation(network_state, a, b, z, seed=None, max_swaps=1, nb_rounds=None):
    """Run Full anticipation mode simulation (simplified, for testing).

    Args:
        max_swaps: Maximum number of simultaneous supplier swaps (1=single, 2=up to dual).
                   A dual swap counts as 2 rewirings.
        nb_rounds: Number of rounds (defaults to module-level NB_ROUNDS).
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = len(network_state['AiSi'])
    Wbar = network_state['Wbar']
    AiSi = network_state['AiSi']
    supplier_id_list = [list(s) for s in network_state['supplier_id_list']]
    alternate_supplier_id_list = [list(s) for s in network_state['alternate_supplier_id_list']]
    W = network_state['W0'].copy()

    # Initial equilibrium
    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
    eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
    initial_utility = calculate_utility(eq)

    _nb_rounds = nb_rounds if nb_rounds is not None else NB_ROUNDS
    for r in range(1, _nb_rounds + 1):
        rewirings_this_round = 0

        for id_firm in np.random.permutation(n):
            current_cost = eq['P'][id_firm]
            potential_cost = current_cost
            best_removes, best_adds = None, None

            current_set = set(supplier_id_list[id_firm])
            alternates = alternate_supplier_id_list[id_firm]

            for swap_size in range(1, max_swaps + 1):
                if len(alternates) < swap_size or len(supplier_id_list[id_firm]) < swap_size:
                    continue
                for new_sups in combinations(alternates, swap_size):
                    for new_s in new_sups:
                        W[new_s, id_firm] = Wbar[new_s, id_firm]

                    for old_sups in combinations(supplier_id_list[id_firm], swap_size):
                        for old_s in old_sups:
                            W[old_s, id_firm] = 0

                        tmp_supplier_list = copy.deepcopy(supplier_id_list)
                        tmp_supplier_list[id_firm] = sorted(
                            (current_set | set(new_sups)) - set(old_sups)
                        )
                        tmp_adjusted_z = compute_adjusted_z(AiSi, tmp_supplier_list, z)
                        new_eq = compute_equilibrium_full(a, b, tmp_adjusted_z, W, n)
                        estimated_cost = new_eq['P'][id_firm]

                        if estimated_cost < potential_cost - EPSILON:
                            potential_cost = estimated_cost
                            best_removes, best_adds = list(old_sups), list(new_sups)

                        for old_s in old_sups:
                            W[old_s, id_firm] = Wbar[old_s, id_firm]

                    for new_s in new_sups:
                        W[new_s, id_firm] = 0

            if best_adds is not None:
                for new_s in best_adds:
                    W[new_s, id_firm] = Wbar[new_s, id_firm]
                for old_s in best_removes:
                    W[old_s, id_firm] = 0

                for old_s, new_s in zip(best_removes, best_adds):
                    supplier_id_list[id_firm].remove(old_s)
                    supplier_id_list[id_firm].append(new_s)
                    alternate_supplier_id_list[id_firm].remove(new_s)
                    alternate_supplier_id_list[id_firm].append(old_s)
                supplier_id_list[id_firm].sort()

                adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
                eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
                rewirings_this_round += len(best_adds)

        if rewirings_this_round == 0:
            break

    return {
        'converged': rewirings_this_round == 0,
        'rounds': r,
        'initial_utility': initial_utility,
        'final_utility': calculate_utility(eq),
        'final_prices': eq['P'],
        'final_supplier_list': supplier_id_list,
    }


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def run_convergence_test(sim_mode="aa", test_mode="same_start_different_order",
                         n_trials=10, network_seed=42, cc=CC, show_comparison=False,
                         max_swaps=1):
    """
    Test convergence properties for AA or Full anticipation mode.

    Args:
        sim_mode: "aa" or "full"
        test_mode: "same_start_different_order" or "different_start_same_parameter"
        n_trials: Number of trials to run
        network_seed: Seed for generating Wbar and AiSi
        cc: Number of alternate suppliers per firm
        show_comparison: Show detailed comparison of mismatches
    """
    n = NB_FIRMS
    same_initial_network = (test_mode == "same_start_different_order")

    print("=" * 70)
    print(f"{sim_mode.upper()} TEST: {test_mode}")
    if same_initial_network:
        print("Same initial network, different permutation orders")
    else:
        print("Different initial networks, same Wbar/AiSi")
    print("=" * 70)
    print(f"Max swaps per rewiring: {max_swaps}")

    # Generate economic parameters for full mode
    if sim_mode == "full":
        random.seed(network_seed)
        np.random.seed(network_seed)
        b = generate_parameter(B_CONFIG, n, 'b', verbose=True)
        a = generate_a_parameter(A_CONFIG, b, n, verbose=True)
        z = generate_parameter(Z_CONFIG, n, 'z', verbose=True)
        print(f"\n  a*b: max={np.max(a * b):.4f} (must be < 1)")
    else:
        a = A_VALUE
        b = None
        z = None

    print(f"\nParameters: n={n}, c={C}, cc={cc}, AiSi_spread={AISI_SPREAD}")
    print(f"Network seed (for Wbar/AiSi): {network_seed}")
    print(f"Number of trials: {n_trials}")

    # Generate base network
    print("\nGenerating Wbar and AiSi...")
    if sim_mode == "full":
        base_state = generate_base_network(n, C, cc, AISI_SPREAD, seed=network_seed,
                                           a=a, b=b, sigma_w=SIGMA_W)
    else:
        base_state = generate_base_network(n, C, cc, AISI_SPREAD, seed=network_seed)

    Wbar = base_state['Wbar']
    AiSi = base_state['AiSi']

    combos_per_firm = [len(AiSi[firm]) for firm in range(n)]
    print(f"  Possible supplier combinations per firm: min={min(combos_per_firm)}, max={max(combos_per_firm)}")

    # Run trials
    print(f"\nRunning {n_trials} trials...")
    print("-" * 70)

    results = []
    for trial in range(n_trials):
        if same_initial_network:
            trial_state = base_state
            perm_seed = 1000 + trial
        else:
            init_seed = 2000 + trial
            trial_state = generate_random_initial_network(n, Wbar, AiSi, seed=init_seed)
            perm_seed = 999

        init_hash = hash(tuple(tuple(sorted(s)) for s in trial_state['supplier_id_list']))

        # Run appropriate simulation
        if sim_mode == "aa":
            result = run_aa_simulation(trial_state, a, seed=perm_seed, max_swaps=max_swaps)
        else:
            result = run_full_simulation(trial_state, a, b, z, seed=perm_seed, max_swaps=max_swaps)

        results.append(result)

        final_hash = hash(tuple(tuple(sorted(s)) for s in result['final_supplier_list']))

        print(f"Trial {trial + 1:2d}: "
              f"init_hash={init_hash % 10000:04d}, "
              f"final_hash={final_hash % 10000:04d}, "
              f"rounds={result['rounds']:2d}, "
              f"utility={result['final_utility']:.6f}, "
              f"price_range=[{result['final_prices'].min():.4f}, {result['final_prices'].max():.4f}]")

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    ref = results[0]
    ref_prices = ref['final_prices']
    ref_suppliers = ref['final_supplier_list']

    prices_match = True
    suppliers_match = True
    max_price_diff = 0

    for i, res in enumerate(results[1:], start=2):
        price_diff = np.max(np.abs(res['final_prices'] - ref_prices))
        max_price_diff = max(max_price_diff, price_diff)
        if price_diff > 1e-10:
            prices_match = False
            if show_comparison:
                print(f"  Trial {i}: Price mismatch! Max diff = {price_diff:.6e}")

        for firm in range(n):
            if sorted(res['final_supplier_list'][firm]) != sorted(ref_suppliers[firm]):
                suppliers_match = False
                if show_comparison:
                    print(f"  Trial {i}: Supplier mismatch for firm {firm}!")
                    print(f"    Reference: {sorted(ref_suppliers[firm])}")
                    print(f"    Trial {i}:  {sorted(res['final_supplier_list'][firm])}")

    print(f"\nMax price difference across trials: {max_price_diff:.6e}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    if prices_match and suppliers_match:
        print("PASS: All trials converged to identical equilibrium")
        print(f"  - Final prices: identical across all {n_trials} trials")
        print(f"  - Final network: identical across all {n_trials} trials")
        if same_initial_network:
            print("  -> Permutation order does NOT affect result")
        else:
            print("  -> Equilibrium is UNIQUE (path-independent)")
    else:
        print("FAIL: Trials converged to different equilibria")
        if not prices_match:
            print("  - Prices differ across trials")
        if not suppliers_match:
            print("  - Network structure differs across trials")
        if same_initial_network:
            print("  -> Permutation order DOES affect result (unexpected!)")
        else:
            print("  -> Equilibrium is NOT unique (path-dependent)")

    return prices_match and suppliers_match


# =============================================================================
# COMPARE AA VS FULL TEST
# =============================================================================

def run_compare_test(test_mode="same_start_different_order", n_trials=10,
                     network_seed=42, cc=CC, show_comparison=False, max_swaps=1):
    """
    Compare AA and Full anticipation modes on the same network(s).

    Uses consistent parameters: a=0.5 (homogeneous), b=1.0, z=1.0

    Args:
        test_mode: "same_start_different_order" or "different_start_same_parameter"
        n_trials: Number of trials to run
        network_seed: Seed for generating Wbar and AiSi
        cc: Number of alternate suppliers per firm
        show_comparison: Show detailed comparison of mismatches
    """
    n = NB_FIRMS
    same_initial_network = (test_mode == "same_start_different_order")

    # Consistent parameters for both modes
    a_scalar = A_VALUE  # 0.5
    a_array = np.full(n, A_VALUE)
    b_array = np.ones(n)
    z_array = np.ones(n)

    print("=" * 70)
    print(f"COMPARE AA vs FULL: {test_mode}")
    if same_initial_network:
        print("Same initial network, different permutation orders")
    else:
        print("Different initial networks, same Wbar/AiSi")
    print("=" * 70)
    print(f"Max swaps per rewiring: {max_swaps}")

    print(f"\nParameters: n={n}, c={C}, cc={cc}, AiSi_spread={AISI_SPREAD}")
    print(f"Economic params: a={A_VALUE}, b=1.0, z=1.0 (homogeneous)")
    print(f"Network seed: {network_seed}")
    print(f"Number of trials: {n_trials}")

    # Generate base network (no a, b for uniform weights)
    print("\nGenerating Wbar and AiSi...")
    random.seed(network_seed)
    np.random.seed(network_seed)
    base_state = generate_base_network(n, C, cc, AISI_SPREAD, seed=network_seed)

    Wbar = base_state['Wbar']
    AiSi = base_state['AiSi']

    combos_per_firm = [len(AiSi[firm]) for firm in range(n)]
    print(f"  Possible supplier combinations per firm: min={min(combos_per_firm)}, max={max(combos_per_firm)}")

    # Run trials
    print(f"\nRunning {n_trials} trials...")
    print("-" * 70)

    all_match = True
    max_price_diff_overall = 0

    for trial in range(n_trials):
        if same_initial_network:
            trial_state = base_state
            perm_seed = 1000 + trial
        else:
            init_seed = 2000 + trial
            trial_state = generate_random_initial_network(n, Wbar, AiSi, seed=init_seed)
            perm_seed = 999

        # Run AA
        result_aa = run_aa_simulation(trial_state, a_scalar, seed=perm_seed, max_swaps=max_swaps)

        # Run Full (need fresh copy of trial_state since simulation modifies it)
        if same_initial_network:
            trial_state_full = base_state
        else:
            trial_state_full = generate_random_initial_network(n, Wbar, AiSi, seed=init_seed)

        result_full = run_full_simulation(trial_state_full, a_array, b_array, z_array, seed=perm_seed,
                                          max_swaps=max_swaps)

        # Compare results
        price_diff = np.max(np.abs(result_aa['final_prices'] - result_full['final_prices']))
        max_price_diff_overall = max(max_price_diff_overall, price_diff)

        suppliers_match = all(
            sorted(result_aa['final_supplier_list'][firm]) == sorted(result_full['final_supplier_list'][firm])
            for firm in range(n)
        )

        match_str = "MATCH" if suppliers_match and price_diff < 1e-10 else "DIFFER"
        if match_str == "DIFFER":
            all_match = False

        print(f"Trial {trial + 1:2d}: "
              f"AA rounds={result_aa['rounds']:2d}, "
              f"Full rounds={result_full['rounds']:2d}, "
              f"price_diff={price_diff:.2e}, "
              f"suppliers={match_str}")

        if show_comparison and match_str == "DIFFER":
            print(f"    AA utility={result_aa['final_utility']:.6f}, "
                  f"Full utility={result_full['final_utility']:.6f}")
            for firm in range(n):
                aa_sup = sorted(result_aa['final_supplier_list'][firm])
                full_sup = sorted(result_full['final_supplier_list'][firm])
                if aa_sup != full_sup:
                    print(f"    Firm {firm}: AA={aa_sup}, Full={full_sup}")

    # Summary
    print("\n" + "=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Max price difference: {max_price_diff_overall:.2e}")

    if all_match:
        print(f"PASS: AA and Full converged to identical equilibria in all {n_trials} trials")
        print("  -> Both algorithms reach the same equilibrium")
    else:
        print(f"FAIL: AA and Full converged to different equilibria in some trials")
        print("  -> Algorithms may reach different local optima")

    return all_match


# =============================================================================
# B-SWEEP: CONVERGENCE AS A FUNCTION OF b
# =============================================================================

def run_b_sweep(b_values=None, test_mode="different_start_same_parameter",
                n_trials=10, network_seed=123, cc=CC, max_swaps=MAX_SWAPS):
    """
    For each homogeneous b value, run n_trials full-mode simulations and report
    convergence diagnostics.

    This helps understand:
    - b < 1 (decreasing returns): expected to converge well
    - b = 1 (constant returns): boundary case
    - b > 1 (increasing returns): may fail to converge or produce cycles
    """
    if b_values is None:
        b_values = B_VALUES
    n = NB_FIRMS
    same_initial_network = (test_mode == "same_start_different_order")

    print("=" * 72)
    print("B-SWEEP CONVERGENCE STUDY  (SIM_MODE=full, COMPARE_MODES=False)")
    print(f"b values: {b_values}")
    print(f"n={n}, c={C}, cc={cc}, AiSi_spread={AISI_SPREAD}, max_swaps={max_swaps}")
    print(f"Rounds={NB_ROUNDS}, Trials={n_trials}, Network seed={network_seed}")
    print(f"Test mode: {test_mode}")
    print("=" * 72)

    summary_table = []

    for b_val in b_values:
        print(f"\n{'=' * 72}")
        print(f"  b = {b_val}  (homogeneous)")
        print(f"{'=' * 72}")

        # Generate economic parameters for this b value
        b_config = {'mode': 'homogeneous', 'value': b_val}
        random.seed(network_seed)
        np.random.seed(network_seed)
        b = generate_parameter(b_config, n, 'b', verbose=True)
        a = generate_a_parameter(A_CONFIG, b, n, verbose=True)
        z = generate_parameter(Z_CONFIG, n, 'z', verbose=True)
        print(f"  a*b: max={np.max(a * b):.4f}, min={np.min(a * b):.4f}")
        print(f"  (1-a)*b: max={np.max((1 - a) * b):.4f}")

        # Generate base network with these parameters
        base_state = generate_base_network(n, C, cc, AISI_SPREAD, seed=network_seed,
                                           a=a, b=b, sigma_w=SIGMA_W)
        Wbar = base_state['Wbar']
        AiSi = base_state['AiSi']

        # Diagnostic: check weight bounds
        active_weights = Wbar[Wbar > 0]
        print(f"  Wbar: min={active_weights.min():.4f}, max={active_weights.max():.4f}, "
              f"mean={active_weights.mean():.4f}")
        # Spectral radius condition: b*(1-a)*w should be < 1
        bw_products = np.array([
            b_val * (1 - a[j]) * Wbar[i, j]
            for j in range(n) for i in range(n) if Wbar[i, j] > 0
        ])
        print(f"  b*(1-a)*w_ij: max={bw_products.max():.4f} (should be < 1 for price eq. stability)")

        combos_per_firm = [len(AiSi[firm]) for firm in range(n)]
        print(f"  Supplier combos per firm: min={min(combos_per_firm)}, max={max(combos_per_firm)}")

        # Run trials
        print(f"\n  Running {n_trials} trials...")
        print(f"  {'-' * 66}")

        results = []
        for trial in range(n_trials):
            if same_initial_network:
                trial_state = base_state
                perm_seed = 1000 + trial
            else:
                init_seed = 2000 + trial
                trial_state = generate_random_initial_network(n, Wbar, AiSi, seed=init_seed)
                perm_seed = 999

            result = run_full_simulation(trial_state, a, b, z, seed=perm_seed,
                                         max_swaps=max_swaps)
            results.append(result)

            conv_tag = "CONV" if result['converged'] else "NO_CONV"
            rel_u = ((result['final_utility'] - result['initial_utility'])
                     / abs(result['initial_utility'])
                     if abs(result['initial_utility']) > 1e-12 else float('nan'))
            print(f"    trial {trial:2d}  {conv_tag:8s}  rounds={result['rounds']:2d}  "
                  f"U0={result['initial_utility']:+.4f}  Uf={result['final_utility']:+.4f}  "
                  f"dU/|U0|={rel_u:+.6f}")

        # Summary stats for this b value
        n_converged = sum(1 for r in results if r['converged'])
        n_not_conv = n_trials - n_converged
        rounds_conv = [r['rounds'] for r in results if r['converged']]
        rounds_all = [r['rounds'] for r in results]
        rel_changes = [
            (r['final_utility'] - r['initial_utility']) / abs(r['initial_utility'])
            for r in results if abs(r['initial_utility']) > 1e-12
        ]

        print(f"\n  SUMMARY for b={b_val}:")
        print(f"    Converged     : {n_converged}/{n_trials}")
        print(f"    Not converged : {n_not_conv}/{n_trials}")
        if rounds_conv:
            print(f"    Rounds (conv) : {np.mean(rounds_conv):.1f} +/- {np.std(rounds_conv):.1f}")
        print(f"    Rounds (all)  : {np.mean(rounds_all):.1f} +/- {np.std(rounds_all):.1f}")
        if rel_changes:
            print(f"    dU/|U0|       : {np.mean(rel_changes):+.6f} +/- {np.std(rel_changes):.6f}")

        # Check uniqueness of final equilibrium (among converged trials)
        converged_results = [r for r in results if r['converged']]
        if len(converged_results) >= 2:
            ref_prices = converged_results[0]['final_prices']
            ref_suppliers = converged_results[0]['final_supplier_list']
            prices_match = True
            suppliers_match = True
            max_price_diff = 0
            for res in converged_results[1:]:
                pdiff = np.max(np.abs(res['final_prices'] - ref_prices))
                max_price_diff = max(max_price_diff, pdiff)
                if pdiff > 1e-8:
                    prices_match = False
                for firm in range(n):
                    if sorted(res['final_supplier_list'][firm]) != sorted(ref_suppliers[firm]):
                        suppliers_match = False
            uniq = "UNIQUE" if (prices_match and suppliers_match) else "MULTIPLE"
            print(f"    Equilibrium   : {uniq} (max price diff={max_price_diff:.2e})")

        summary_table.append({
            'b': b_val,
            'converged': n_converged,
            'not_converged': n_not_conv,
            'mean_rounds': np.mean(rounds_all),
            'mean_dU': np.mean(rel_changes) if rel_changes else float('nan'),
        })

    # Final comparison table
    print(f"\n{'=' * 72}")
    print("COMPARISON TABLE")
    print(f"{'=' * 72}")
    print(f"  {'b':>5s}  {'conv':>5s}  {'no_conv':>7s}  {'rounds':>8s}  {'dU/|U0|':>12s}")
    print(f"  {'-' * 5}  {'-' * 5}  {'-' * 7}  {'-' * 8}  {'-' * 12}")
    for row in summary_table:
        print(f"  {row['b']:5.2f}  {row['converged']:5d}  {row['not_converged']:7d}  "
              f"{row['mean_rounds']:8.1f}  {row['mean_dU']:+12.6f}")
    print(f"{'=' * 72}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Start with configured values
    sim_mode = SIM_MODE
    test_mode = TEST_MODE
    compare_modes = COMPARE_MODES
    cc = CC
    n_trials = N_TRIALS
    network_seed = NETWORK_SEED
    show_comparison = SHOW_COMPARISON
    max_swaps = MAX_SWAPS
    b_sweep_mode = True  # Default: run b-sweep study

    # Command line overrides
    for arg in sys.argv[1:]:
        if arg == "--aa":
            sim_mode = "aa"
            b_sweep_mode = False
        elif arg == "--full":
            sim_mode = "full"
            b_sweep_mode = False
        elif arg == "--compare":
            compare_modes = True
            b_sweep_mode = False
        elif arg == "--b_sweep":
            b_sweep_mode = True
        elif arg == "--same_start_different_order":
            test_mode = "same_start_different_order"
        elif arg == "--different_start_same_parameter":
            test_mode = "different_start_same_parameter"
        elif arg.startswith("--cc="):
            cc = int(arg.split("=")[1])
        elif arg.startswith("--trials="):
            n_trials = int(arg.split("=")[1])
        elif arg.startswith("--seed="):
            network_seed = int(arg.split("=")[1])
        elif arg == "--show":
            show_comparison = True
        elif arg.startswith("--max_swaps="):
            max_swaps = int(arg.split("=")[1])

    if b_sweep_mode:
        run_b_sweep(
            b_values=B_VALUES,
            test_mode=test_mode,
            n_trials=n_trials,
            network_seed=network_seed,
            cc=cc,
            max_swaps=max_swaps,
        )
    elif compare_modes:
        success = run_compare_test(
            test_mode=test_mode,
            n_trials=n_trials,
            network_seed=network_seed,
            cc=cc,
            show_comparison=show_comparison,
            max_swaps=max_swaps,
        )
        exit(0 if success else 1)
    else:
        success = run_convergence_test(
            sim_mode=sim_mode,
            test_mode=test_mode,
            n_trials=n_trials,
            network_seed=network_seed,
            cc=cc,
            show_comparison=show_comparison,
            max_swaps=max_swaps,
        )
        exit(0 if success else 1)
