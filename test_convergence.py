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

Compare swaps mode (--compare_swaps):
- For each max_swaps value, runs n_trials from different starting points
- Same technology matrix (Wbar, AiSi) shared across all max_swaps values
- Measures diversity of equilibria within each max_swaps group
- Cross-compares: does higher max_swaps improve utility / reduce diversity?
- Default: compare max_swaps=1 vs 2; override with --swap_values=1,2,3
- Works with both AA (--aa) and Full (--full) simulation modes

Sweep mode (--sweep):
- Loops over all combos: b in {0.9, 1.0, 1.1, U(0.8,1.2)}, cc in {1,2,3,4}, max_swaps in {1,2}
- For each combo, runs n_trials from different starting networks
- Reports convergence, uniqueness, utility spread in a summary table
- Usage: python test_convergence.py --aa --sweep --trials=10
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
    identify_firms_within_tier_np,
    compute_partial_equilibrium_cost,
    compute_partial_equilibrium_cost_naive,
)


# =============================================================================
# TEST CONFIGURATION (modify here to run without command line)
# =============================================================================

SIM_MODE = "async_aa"  # "aa" or "full"
TEST_MODE = "different_start_same_parameter"  # "same_start_different_order" or "different_start_same_parameter"
COMPARE_MODES = False  # If True, compare AA vs Full instead of running single mode
COMPARE_SWAPS = False  # If True, compare max_swaps=1 vs max_swaps=2 on same network(s)

# Shared parameters
NB_FIRMS = 100
NB_ROUNDS = 20
C = 4
CC = 4
AISI_SPREAD = 0.1
MAX_SWAPS = 1  # max number of simultaneous supplier swaps (1 = single, 2 = up to dual)

# AA mode parameters
A_VALUE = 0.5  # homogeneous labor share for AA mode

# Full mode parameters (can be homogeneous or heterogeneous)
B_CONFIG = {'mode': 'homogeneous', 'value': 0.9}
# B_CONFIG = {'mode': 'uniform', 'min': 0.5, "max": 1.5}
A_CONFIG = {'mode': 'homogeneous', 'value': 0.5}
Z_CONFIG = {'mode': 'homogeneous', 'value': 1.0}
# Z_CONFIG = {'mode': 'uniform', 'min': 0.5, "max": 1.5}
SIGMA_W = 0.0

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

def run_aa_simulation(network_state, a, seed=None, max_swaps=1, nb_rounds=None,
                      b=None, z=None):
    """Run AA mode simulation (simplified, for testing).

    Firms evaluate supplier swaps using frozen reference prices (AA principle):
        cost = prod(P_ref^((1-a)*W_col)) / z_i

    After all firms decide, swaps are applied and equilibrium is recomputed.

    When b is None (CRS mode, b=1): uses compute_equilibrium_crs, backward compatible.
    When b is provided (non-CRS): uses compute_equilibrium_full with true equilibrium
    recomputation at the end of each round.

    Args:
        network_state: Network state dict from generate_base_network.
        a: Labor share (scalar for CRS, array for non-CRS).
        seed: Random seed for permutation order.
        max_swaps: Maximum number of simultaneous supplier swaps (1=single, 2=up to dual).
        nb_rounds: Number of rounds (defaults to module-level NB_ROUNDS).
        b: Returns to scale array (None for CRS mode).
        z: Base productivity array (None for CRS mode).
    """
    crs_mode = b is None

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
    if crs_mode:
        adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
        eq = compute_equilibrium_crs(a, adjusted_z, W, n)
        initial_utility = -eq['P'].sum()
    else:
        adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
        eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
        initial_utility = calculate_utility(eq)

    P_ref = eq['P'].copy()

    # For the decision formula, we need scalar (1-a) per firm
    # In CRS mode a is scalar; in non-CRS mode a is array
    if crs_mode:
        one_minus_a = (1 - a)  # scalar
    else:
        one_minus_a = (1 - a)  # array, indexed per firm below

    _nb_rounds = nb_rounds if nb_rounds is not None else NB_ROUNDS
    for r in range(1, _nb_rounds + 1):
        aa_do = np.zeros(n, dtype=bool)
        aa_removes = [None] * n
        aa_adds = [None] * n

        for id_firm in np.random.permutation(n):
            # Per-firm (1-a) factor
            oma = one_minus_a if crs_mode else one_minus_a[id_firm]

            current_W_col = np.zeros(n)
            for s in supplier_id_list[id_firm]:
                current_W_col[s] = Wbar[s, id_firm]
            current_z = AiSi[id_firm][tuple(int(s) for s in sorted(supplier_id_list[id_firm]))]
            current_cost = np.prod(np.power(P_ref, oma * current_W_col)) / current_z
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
                        cost = np.prod(np.power(P_ref, oma * W_col)) / test_z

                        if cost < potential_cost - EPSILON:
                            potential_cost = cost
                            best_removes, best_adds = list(old_sups), list(new_sups)

            if best_adds is not None:
                aa_do[id_firm] = True
                aa_removes[id_firm] = best_removes
                aa_adds[id_firm] = best_adds

        # Count rewirings
        rewirings_this_round = sum(1 for d in aa_do if d)

        # Apply swaps
        for id_firm in range(n):
            if aa_do[id_firm]:
                for old_s, new_s in zip(aa_removes[id_firm], aa_adds[id_firm]):
                    supplier_id_list[id_firm].remove(old_s)
                    supplier_id_list[id_firm].append(new_s)
                    alternate_supplier_id_list[id_firm].remove(new_s)
                    alternate_supplier_id_list[id_firm].append(old_s)
                supplier_id_list[id_firm].sort()

        W = build_W_from_suppliers(supplier_id_list, Wbar)

        # Convergence: no rewiring happened
        if rewirings_this_round == 0:
            break

        # Recompute true equilibrium for new network
        if crs_mode:
            adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
            eq = compute_equilibrium_crs(a, adjusted_z, W, n)
        else:
            adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
            eq = compute_equilibrium_full(a, b, adjusted_z, W, n)

        P_ref = eq['P'].copy()

    # Final equilibrium (if we broke out on round 1 with no swaps, eq is already current)
    if crs_mode:
        final_z = compute_adjusted_z(AiSi, supplier_id_list)
        final_eq = compute_equilibrium_crs(a, final_z, W, n)
        final_utility = -final_eq['P'].sum()
    else:
        final_eq = eq  # already computed at end of last round
        final_utility = calculate_utility(final_eq)

    return {
        'converged': rewirings_this_round == 0,
        'rounds': r,
        'initial_utility': initial_utility,
        'final_utility': final_utility,
        'final_prices': final_eq['P'],
        'final_supplier_list': supplier_id_list,
    }


# =============================================================================
# ASYNC AA MODE SIMULATION
# =============================================================================

def run_async_aa_simulation(network_state, a, seed=None, max_swaps=1, nb_rounds=None,
                            b=None, z=None):
    """Run AA mode simulation with ASYNCHRONOUS (Gauss-Seidel) updates.

    Same cost formula as run_aa_simulation:
        cost = prod(P_ref^((1-a)*W_col)) / z_i

    but the swap is applied immediately after each firm decides, and the true
    equilibrium is recomputed right away. The next firm in the permutation uses
    the updated P_ref. Convergence criterion: no rewirings during a full round.

    Args:
        network_state: Network state dict from generate_base_network.
        a: Labor share (scalar for CRS, array for non-CRS).
        seed: Random seed for permutation order.
        max_swaps: Maximum number of simultaneous supplier swaps.
        nb_rounds: Number of rounds (defaults to module-level NB_ROUNDS).
        b: Returns to scale array (None for CRS mode).
        z: Base productivity array (None for CRS mode).
    """
    crs_mode = b is None

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
    if crs_mode:
        adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
        eq = compute_equilibrium_crs(a, adjusted_z, W, n)
        initial_utility = -eq['P'].sum()
    else:
        adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
        eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
        initial_utility = calculate_utility(eq)

    P_ref = eq['P'].copy()

    if crs_mode:
        one_minus_a = (1 - a)
    else:
        one_minus_a = (1 - a)

    _nb_rounds = nb_rounds if nb_rounds is not None else NB_ROUNDS
    rewirings_this_round = 0
    for r in range(1, _nb_rounds + 1):
        rewirings_this_round = 0

        for id_firm in np.random.permutation(n):
            oma = one_minus_a if crs_mode else one_minus_a[id_firm]

            current_W_col = np.zeros(n)
            for s in supplier_id_list[id_firm]:
                current_W_col[s] = Wbar[s, id_firm]
            current_z = AiSi[id_firm][tuple(int(s) for s in sorted(supplier_id_list[id_firm]))]
            current_cost = np.prod(np.power(P_ref, oma * current_W_col)) / current_z
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
                        cost = np.prod(np.power(P_ref, oma * W_col)) / test_z

                        if cost < potential_cost - EPSILON:
                            potential_cost = cost
                            best_removes, best_adds = list(old_sups), list(new_sups)

            if best_adds is not None:
                for old_s, new_s in zip(best_removes, best_adds):
                    supplier_id_list[id_firm].remove(old_s)
                    supplier_id_list[id_firm].append(new_s)
                    alternate_supplier_id_list[id_firm].remove(new_s)
                    alternate_supplier_id_list[id_firm].append(old_s)
                supplier_id_list[id_firm].sort()

                W = build_W_from_suppliers(supplier_id_list, Wbar)

                if crs_mode:
                    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
                    eq = compute_equilibrium_crs(a, adjusted_z, W, n)
                else:
                    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
                    eq = compute_equilibrium_full(a, b, adjusted_z, W, n)

                P_ref = eq['P'].copy()
                rewirings_this_round += len(best_adds)

        if rewirings_this_round == 0:
            break

    if crs_mode:
        final_z = compute_adjusted_z(AiSi, supplier_id_list)
        final_eq = compute_equilibrium_crs(a, final_z, W, n)
        final_utility = -final_eq['P'].sum()
    else:
        final_eq = eq
        final_utility = calculate_utility(final_eq)

    return {
        'converged': rewirings_this_round == 0,
        'rounds': r,
        'initial_utility': initial_utility,
        'final_utility': final_utility,
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
# UNIFIED ASYNC SIMULATION (aa / full / limited)
# =============================================================================

def _edges_from_suppliers(supplier_id_list):
    """Return (K, 2) int32 array of (supplier, buyer) pairs, lexicographically sorted."""
    pairs = [(int(s), int(buyer))
             for buyer, suppliers in enumerate(supplier_id_list)
             for s in suppliers]
    pairs.sort()
    return np.asarray(pairs, dtype=np.int32)


def run_unified_simulation(network_state, a, b, z, mode="aa", seed=None,
                           max_swaps=1, nb_rounds=None, tier=None, trace=False):
    """Unified asynchronous rewiring simulation.

    Common structure for all three anticipation modes:
    - Round = one permutation pass over firms.
    - Each firm evaluates all combinations up to max_swaps against its current
      supplier set, using a mode-specific cost function.
    - If the best candidate strictly reduces the (mode-specific) cost, the swap
      is applied IMMEDIATELY, the true GE is recomputed, and the next firm in
      the permutation sees the updated state.
    - Convergence: no swaps during a full round.

    Mode differs only in evaluate_candidate_cost:
    - 'aa': ratchet / closed-form — prod(P^((1-a_i)*W_col')) / test_z
            where P is the current equilibrium price vector.
    - 'full': solve compute_equilibrium_full on the hypothetical W (current W
              with firm i's column replaced); take the i-th price.
    - 'limited': identify firms within tier[i] hops on the hypothetical W's
                 adjacency; solve partial equilibrium on that subgraph; take
                 the i-th price.

    Args:
        network_state: dict from generate_base_network.
        a, b, z: economic parameter arrays of length n.
        mode: 'aa', 'full', 'limited' (boundary-corrected partial equilibrium),
              or 'naive_limited' (legacy: drop links to outside firms).
        seed: random seed for permutation order.
        max_swaps: max simultaneous supplier swaps (1=single, 2=up to dual).
        nb_rounds: number of rounds (defaults to module-level NB_ROUNDS).
        tier: int or np.ndarray of length n. Required for 'limited' / 'naive_limited'.
              If int, broadcast to all firms.
        trace: if True, collect per-round scalars and edge snapshots and attach
               them to the result dict as result['trace'] = {
                   'scalars': list[dict],  # one entry per round-end, keys t, rewirings, sum_p, max_swap_binding
                   'edges':   list[np.ndarray],  # length rounds+1, each (N*mean_cc, 2) int32
                   'converged_at': int or None,
               }
    """
    if mode not in ("aa", "full", "limited", "naive_limited"):
        raise ValueError(
            f"Unknown mode: {mode!r}. Use 'aa', 'full', 'limited', or 'naive_limited'."
        )
    if mode in ("limited", "naive_limited") and tier is None:
        raise ValueError(f"tier must be provided for mode={mode!r}.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    n = len(network_state['AiSi'])
    Wbar = network_state['Wbar']
    AiSi = network_state['AiSi']
    supplier_id_list = [list(s) for s in network_state['supplier_id_list']]
    alternate_supplier_id_list = [list(s) for s in network_state['alternate_supplier_id_list']]
    W = network_state['W0'].copy()

    # Broadcast tier if scalar
    if mode in ("limited", "naive_limited"):
        tier_arr = np.full(n, int(tier)) if np.isscalar(tier) else np.asarray(tier, dtype=int)

    # Initial equilibrium
    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
    eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
    initial_utility = calculate_utility(eq)

    one_minus_a = 1 - a

    def candidate_cost(id_firm, candidate_set):
        """Cost of id_firm if its supplier set were `candidate_set`, under mode."""
        # Hypothetical column for this firm
        W_col = np.zeros(n)
        for s in candidate_set:
            W_col[s] = Wbar[s, id_firm]
        test_z = AiSi[id_firm][tuple(sorted(int(s) for s in candidate_set))]

        if mode == "aa":
            return float(np.prod(np.power(eq['P'], one_minus_a[id_firm] * W_col)) / test_z)

        # For full and limited variants: build W_test = current W with firm i's column replaced
        W_test = W.copy()
        W_test[:, id_firm] = W_col

        # Build tmp supplier list and adjusted z for the hypothetical network
        tmp_supplier_list = [list(s) for s in supplier_id_list]
        tmp_supplier_list[id_firm] = sorted(int(s) for s in candidate_set)
        tmp_adjusted_z = compute_adjusted_z(AiSi, tmp_supplier_list, z)

        if mode == "full":
            new_eq = compute_equilibrium_full(a, b, tmp_adjusted_z, W_test, n)
            return float(new_eq['P'][id_firm])

        # tier neighborhood on the HYPOTHETICAL graph (used by both limited variants)
        M_test = (W_test != 0).astype(np.int8)
        firms_within_tiers = identify_firms_within_tier_np(M_test, id_firm, tier_arr[id_firm])

        if mode == "limited":
            # Boundary-corrected partial equilibrium (manuscript spec)
            return compute_partial_equilibrium_cost(
                a, b, tmp_adjusted_z, W, W_test, eq,
                firms_within_tiers, id_firm, n,
            )

        # mode == "naive_limited": drop links to outside, solve on the island only
        return compute_partial_equilibrium_cost_naive(
            a, b, tmp_adjusted_z, W_test, firms_within_tiers, id_firm,
        )

    _nb_rounds = nb_rounds if nb_rounds is not None else NB_ROUNDS

    if trace:
        trace_scalars = [{
            't': 0,
            'rewirings': 0,
            'sum_p': float(eq['P'].sum()),
            'max_swap_binding': False,
        }]
        trace_edges = [_edges_from_suppliers(supplier_id_list)]
        trace_prices = [eq['P'].copy()]
        trace_price_steps = [0]
        trace_rewire_events = []
        converged_at = None

    t = 0
    rewirings_this_round = 0
    for r in range(1, _nb_rounds + 1):
        rewirings_this_round = 0
        max_swap_binding = False

        for id_firm in np.random.permutation(n):
            t += 1
            current_set = set(supplier_id_list[id_firm])
            alternates = alternate_supplier_id_list[id_firm]

            current_cost = candidate_cost(id_firm, current_set)
            potential_cost = current_cost
            best_removes, best_adds = None, None

            for swap_size in range(1, max_swaps + 1):
                if len(alternates) < swap_size or len(current_set) < swap_size:
                    continue
                for new_sups in combinations(alternates, swap_size):
                    for old_sups in combinations(current_set, swap_size):
                        new_set = (current_set - set(old_sups)) | set(new_sups)
                        cost = candidate_cost(id_firm, new_set)
                        if cost < potential_cost - EPSILON:
                            potential_cost = cost
                            best_removes, best_adds = list(old_sups), list(new_sups)

            if best_adds is not None:
                for old_s, new_s in zip(best_removes, best_adds):
                    supplier_id_list[id_firm].remove(old_s)
                    supplier_id_list[id_firm].append(new_s)
                    alternate_supplier_id_list[id_firm].remove(new_s)
                    alternate_supplier_id_list[id_firm].append(old_s)
                supplier_id_list[id_firm].sort()

                W = build_W_from_suppliers(supplier_id_list, Wbar)
                adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
                eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
                rewirings_this_round += len(best_adds)
                if len(best_adds) == max_swaps:
                    max_swap_binding = True
                if trace:
                    trace_rewire_events.append({'t': t, 'round': r, 'firm': int(id_firm)})
                    trace_prices.append(eq['P'].copy())
                    trace_price_steps.append(t)

        if trace:
            trace_scalars.append({
                't': r,
                'rewirings': int(rewirings_this_round),
                'sum_p': float(eq['P'].sum()),
                'max_swap_binding': bool(max_swap_binding),
            })
            trace_edges.append(_edges_from_suppliers(supplier_id_list))

        if rewirings_this_round == 0:
            if trace:
                converged_at = r
            break

    if trace:
        # Anchor the last observation at the final step so the price trajectory
        # does not visually end at the last rewire but extends to t_final.
        if not trace_price_steps or trace_price_steps[-1] != t:
            trace_prices.append(eq['P'].copy())
            trace_price_steps.append(t)

    result = {
        'converged': rewirings_this_round == 0,
        'rounds': r,
        'initial_utility': initial_utility,
        'final_utility': calculate_utility(eq),
        'final_prices': eq['P'],
        'final_supplier_list': supplier_id_list,
        'mode': mode,
    }
    if trace:
        result['trace'] = {
            'scalars': trace_scalars,
            'edges': trace_edges,
            'prices': trace_prices,
            'price_steps': trace_price_steps,
            'rewire_events': trace_rewire_events,
            'converged_at': converged_at,
        }
    return result


# =============================================================================
# MAIN TEST FUNCTION
# =============================================================================

def run_convergence_test(sim_mode="aa", test_mode="same_start_different_order",
                         n_trials=10, network_seed=42, cc=CC, show_comparison=False,
                         max_swaps=1, b_config=None, a_config=None, z_config=None,
                         quiet=False):
    """
    Test convergence properties for AA or Full anticipation mode.

    Args:
        sim_mode: "aa" or "full"
        test_mode: "same_start_different_order" or "different_start_same_parameter"
        n_trials: Number of trials to run
        network_seed: Seed for generating Wbar and AiSi
        cc: Number of alternate suppliers per firm
        show_comparison: Show detailed comparison of mismatches
        b_config: Override B_CONFIG dict (default: use module-level B_CONFIG)
        a_config: Override A_CONFIG dict (default: use module-level A_CONFIG)
        z_config: Override Z_CONFIG dict (default: use module-level Z_CONFIG)
        quiet: If True, suppress per-trial output (for sweep mode)

    Returns:
        dict with keys: 'all_match', 'all_converged', 'n_distinct',
              'n_converged', 'max_price_diff', 'utilities'
    """
    _b_config = b_config if b_config is not None else B_CONFIG
    _a_config = a_config if a_config is not None else A_CONFIG
    _z_config = z_config if z_config is not None else Z_CONFIG

    n = NB_FIRMS
    same_initial_network = (test_mode == "same_start_different_order")

    if not quiet:
        print("=" * 70)
        print(f"{sim_mode.upper()} TEST: {test_mode}")
        if same_initial_network:
            print("Same initial network, different permutation orders")
        else:
            print("Different initial networks, same Wbar/AiSi")
        print("=" * 70)
        print(f"Max swaps per rewiring: {max_swaps}")

    # Generate economic parameters
    random.seed(network_seed)
    np.random.seed(network_seed)
    b = generate_parameter(_b_config, n, 'b', verbose=not quiet)
    a = generate_a_parameter(_a_config, b, n, verbose=not quiet)
    z = generate_parameter(_z_config, n, 'z', verbose=not quiet)
    if not quiet:
        print(f"\n  a*b: max={np.max(a * b):.4f} (must be < 1)")

    if not quiet:
        print(f"\nParameters: n={n}, c={C}, cc={cc}, AiSi_spread={AISI_SPREAD}")
        print(f"Network seed (for Wbar/AiSi): {network_seed}")
        print(f"Number of trials: {n_trials}")

    # Generate base network
    if not quiet:
        print("\nGenerating Wbar and AiSi...")
    base_state = generate_base_network(n, C, cc, AISI_SPREAD, seed=network_seed,
                                       a=a, b=b, sigma_w=SIGMA_W)

    Wbar = base_state['Wbar']
    AiSi = base_state['AiSi']

    combos_per_firm = [len(AiSi[firm]) for firm in range(n)]
    if not quiet:
        print(f"  Possible supplier combinations per firm: min={min(combos_per_firm)}, max={max(combos_per_firm)}")

    # Run trials
    if not quiet:
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
            result = run_aa_simulation(trial_state, a, seed=perm_seed, max_swaps=max_swaps,
                                       b=b, z=z)
        elif sim_mode == "async_aa":
            result = run_async_aa_simulation(trial_state, a, seed=perm_seed, max_swaps=max_swaps,
                                             b=b, z=z)
        else:
            result = run_full_simulation(trial_state, a, b, z, seed=perm_seed, max_swaps=max_swaps)

        results.append(result)

        final_hash = hash(tuple(tuple(sorted(s)) for s in result['final_supplier_list']))

        if not quiet:
            print(f"Trial {trial + 1:2d}: "
                  f"init_hash={init_hash % 10000:04d}, "
                  f"final_hash={final_hash % 10000:04d}, "
                  f"rounds={result['rounds']:2d}, "
                  f"utility={result['final_utility']:.6f}, "
                  f"price_range=[{result['final_prices'].min():.4f}, {result['final_prices'].max():.4f}]")

    # Compare results
    n_converged = sum(1 for r in results if r['converged'])
    all_converged = (n_converged == n_trials)

    # Count distinct equilibria by supplier hash
    hashes = [
        hash(tuple(tuple(sorted(s)) for s in r['final_supplier_list']))
        for r in results
    ]
    n_distinct = len(set(hashes))
    utilities = np.array([r['final_utility'] for r in results])

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
            if show_comparison and not quiet:
                print(f"  Trial {i}: Price mismatch! Max diff = {price_diff:.6e}")

        for firm in range(n):
            if sorted(res['final_supplier_list'][firm]) != sorted(ref_suppliers[firm]):
                suppliers_match = False
                if show_comparison and not quiet:
                    print(f"  Trial {i}: Supplier mismatch for firm {firm}!")
                    print(f"    Reference: {sorted(ref_suppliers[firm])}")
                    print(f"    Trial {i}:  {sorted(res['final_supplier_list'][firm])}")

    all_match = prices_match and suppliers_match

    if not quiet:
        print("\n" + "=" * 70)
        print("COMPARISON")
        print("=" * 70)
        print(f"Max price difference across trials: {max_price_diff:.6e}")

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"Converged: {n_converged}/{n_trials}")
        print(f"Distinct equilibria: {n_distinct}/{n_trials}")

        if all_match:
            print("PASS: All trials converged to identical equilibrium")
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

    return {
        'all_match': all_match,
        'all_converged': all_converged,
        'n_converged': n_converged,
        'n_distinct': n_distinct,
        'max_price_diff': max_price_diff,
        'utilities': utilities,
    }


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
# COMPARE SWAPS TEST
# =============================================================================

def run_compare_swaps(sim_mode="full", test_mode="different_start_same_parameter",
                      n_trials=10, network_seed=42, cc=CC, show_comparison=False,
                      swap_values=None):
    """
    Assess how the diversity of equilibria changes with max_swaps.

    For each max_swaps value, runs n_trials simulations from different starting
    networks (or different permutation orders) using the SAME technology matrix
    (Wbar, AiSi, z, a, b). Then measures within-group diversity: how many
    distinct equilibria, spread of utilities, and pairwise price differences.

    Args:
        sim_mode: "aa" or "full"
        test_mode: "same_start_different_order" or "different_start_same_parameter"
        n_trials: Number of trials per max_swaps value
        network_seed: Seed for generating Wbar and AiSi
        cc: Number of alternate suppliers per firm
        show_comparison: Show per-trial details
        swap_values: List of max_swaps values to compare (default [1, 2])
    """
    if swap_values is None:
        swap_values = [1, 2]

    n = NB_FIRMS
    same_initial_network = (test_mode == "same_start_different_order")

    print("=" * 70)
    print(f"COMPARE SWAPS DIVERSITY — {sim_mode.upper()}: {test_mode}")
    print(f"max_swaps values: {swap_values}")
    if same_initial_network:
        print("Same initial network, different permutation orders")
    else:
        print("Different initial networks, same Wbar/AiSi")
    print("=" * 70)

    # Generate economic parameters (shared across all max_swaps values)
    random.seed(network_seed)
    np.random.seed(network_seed)
    b = generate_parameter(B_CONFIG, n, 'b', verbose=True)
    a = generate_a_parameter(A_CONFIG, b, n, verbose=True)
    z = generate_parameter(Z_CONFIG, n, 'z', verbose=True)
    print(f"\n  a*b: max={np.max(a * b):.4f} (must be < 1)")

    print(f"\nParameters: n={n}, c={C}, cc={cc}, AiSi_spread={AISI_SPREAD}")
    print(f"Network seed (for Wbar/AiSi): {network_seed}")
    print(f"Trials per max_swaps: {n_trials}")

    # Generate base network (shared across all max_swaps values)
    print("\nGenerating Wbar and AiSi...")
    base_state = generate_base_network(n, C, cc, AISI_SPREAD, seed=network_seed,
                                       a=a, b=b, sigma_w=SIGMA_W)

    Wbar = base_state['Wbar']
    AiSi = base_state['AiSi']

    combos_per_firm = [len(AiSi[firm]) for firm in range(n)]
    print(f"  Supplier combinations per firm: min={min(combos_per_firm)}, max={max(combos_per_firm)}")

    # Pre-generate trial states (same across all max_swaps values)
    trial_configs = []
    for trial in range(n_trials):
        if same_initial_network:
            perm_seed = 1000 + trial
            trial_configs.append(('base', perm_seed))
        else:
            init_seed = 2000 + trial
            perm_seed = 999
            trial_configs.append((init_seed, perm_seed))

    # Run all max_swaps x trials
    # results_by_ms[ms_val] = list of result dicts
    results_by_ms = {}

    for ms in swap_values:
        print(f"\n{'─' * 70}")
        print(f"  max_swaps = {ms}")
        print(f"{'─' * 70}")

        results = []
        for trial, (state_key, perm_seed) in enumerate(trial_configs):
            if state_key == 'base':
                trial_state = base_state
            else:
                trial_state = generate_random_initial_network(n, Wbar, AiSi, seed=state_key)

            if sim_mode == "aa":
                res = run_aa_simulation(trial_state, a, seed=perm_seed, max_swaps=ms,
                                        b=b, z=z)
            elif sim_mode == "async_aa":
                res = run_async_aa_simulation(trial_state, a, seed=perm_seed, max_swaps=ms,
                                              b=b, z=z)
            else:
                res = run_full_simulation(trial_state, a, b, z, seed=perm_seed, max_swaps=ms)

            results.append(res)

            if show_comparison:
                final_hash = hash(tuple(tuple(sorted(s)) for s in res['final_supplier_list']))
                print(f"  Trial {trial + 1:2d}: rounds={res['rounds']:2d}, "
                      f"utility={res['final_utility']:.6f}, "
                      f"hash={final_hash % 10000:04d}")

        results_by_ms[ms] = results

    # =========================================================================
    # DIVERSITY ANALYSIS
    # =========================================================================
    print("\n" + "=" * 70)
    print("DIVERSITY ANALYSIS")
    print("=" * 70)

    for ms in swap_values:
        results = results_by_ms[ms]
        utilities = np.array([r['final_utility'] for r in results])

        # Count distinct equilibria by supplier hash
        hashes = [
            hash(tuple(tuple(sorted(s)) for s in r['final_supplier_list']))
            for r in results
        ]
        unique_hashes = set(hashes)
        n_distinct = len(unique_hashes)

        # Pairwise max price differences
        pairwise_diffs = []
        for i, j in combinations(range(len(results)), 2):
            diff = np.max(np.abs(results[i]['final_prices'] - results[j]['final_prices']))
            pairwise_diffs.append(diff)
        pairwise_diffs = np.array(pairwise_diffs)

        print(f"\n  max_swaps = {ms}:")
        print(f"    Distinct equilibria:  {n_distinct} / {n_trials}")
        print(f"    Utility:  mean={utilities.mean():.6f}, std={utilities.std():.6f}, "
              f"range=[{utilities.min():.6f}, {utilities.max():.6f}]")
        if len(pairwise_diffs) > 0:
            print(f"    Pairwise max|Δp|:  mean={pairwise_diffs.mean():.2e}, "
                  f"max={pairwise_diffs.max():.2e}, "
                  f"fraction_identical={np.mean(pairwise_diffs < 1e-10):.1%}")

    # Cross-comparison: for each trial, does higher max_swaps improve utility?
    if len(swap_values) >= 2:
        print(f"\n{'─' * 70}")
        print("CROSS-COMPARISON (per trial, across max_swaps values)")
        print(f"{'─' * 70}")

        ms_min, ms_max = min(swap_values), max(swap_values)
        res_lo = results_by_ms[ms_min]
        res_hi = results_by_ms[ms_max]

        n_better = 0
        n_same = 0
        n_worse = 0
        util_diffs = []

        for trial in range(n_trials):
            u_lo = res_lo[trial]['final_utility']
            u_hi = res_hi[trial]['final_utility']
            diff = u_hi - u_lo
            util_diffs.append(diff)

            sup_match = all(
                sorted(res_lo[trial]['final_supplier_list'][f]) ==
                sorted(res_hi[trial]['final_supplier_list'][f])
                for f in range(n)
            )

            if sup_match:
                n_same += 1
            elif diff > EPSILON:
                n_better += 1
            else:
                n_worse += 1

        util_diffs = np.array(util_diffs)
        print(f"\n  ms={ms_max} vs ms={ms_min} (per trial, same starting network):")
        print(f"    Same equilibrium:     {n_same:3d} / {n_trials}")
        print(f"    ms={ms_max} strictly better: {n_better:3d} / {n_trials}")
        print(f"    ms={ms_max} strictly worse:  {n_worse:3d} / {n_trials}")
        print(f"    Utility diff (ms={ms_max} - ms={ms_min}): "
              f"mean={util_diffs.mean():+.6f}, std={util_diffs.std():.6f}")

    return results_by_ms


# =============================================================================
# CONFIG PARSING UTILITY
# =============================================================================

def parse_config_arg(s):
    """Parse a config string like 'homogeneous:0.5' or 'uniform:0.5:1.5' into a dict."""
    parts = s.split(':')
    mode = parts[0]
    if mode == 'homogeneous':
        return {'mode': 'homogeneous', 'value': float(parts[1])}
    elif mode == 'uniform':
        return {'mode': 'uniform', 'min': float(parts[1]), 'max': float(parts[2])}
    else:
        raise ValueError(f"Unknown config mode: {mode}")


def config_label(cfg):
    """Short label for a config dict, for display in sweep tables."""
    if cfg['mode'] == 'homogeneous':
        return f"{cfg['value']}"
    elif cfg['mode'] == 'uniform':
        return f"U({cfg['min']},{cfg['max']})"
    return str(cfg)


# =============================================================================
# SWEEP MODE
# =============================================================================

def run_sweep(sim_mode="aa", n_trials=10, network_seed=42, show_comparison=False):
    """
    Sweep over parameter combinations and report convergence + uniqueness.

    Tests all combos of:
    - b: 0.9, 1.0, 1.1, uniform(0.8, 1.2)
    - cc: 1, 2, 3, 4
    - max_swaps: 1, 2

    For each combo, runs n_trials from different starting networks
    (different_start_same_parameter) and reports:
    - Whether all trials converged
    - Number of distinct equilibria
    - Utility spread

    Args:
        sim_mode: "aa" or "full"
        n_trials: Number of trials per combo
        network_seed: Base seed for Wbar/AiSi generation
        show_comparison: Show per-trial detail for mismatches
    """
    b_configs = [
        {'mode': 'homogeneous', 'value': 0.9},
        {'mode': 'homogeneous', 'value': 1.0},
        {'mode': 'homogeneous', 'value': 1.1},
        {'mode': 'uniform', 'min': 0.8, 'max': 1.2},
    ]
    cc_values = [1, 2, 3, 4]
    max_swaps_values = [1, 2]

    total_combos = len(b_configs) * len(cc_values) * len(max_swaps_values)

    print("=" * 90)
    print(f"SWEEP — {sim_mode.upper()} mode")
    print(f"Trials per combo: {n_trials}, Base seed: {network_seed}")
    print(f"Total combos: {total_combos}")
    print("=" * 90)

    # Header
    print(f"\n{'b':>12s}  {'cc':>3s}  {'ms':>3s}  "
          f"{'conv':>5s}  {'distinct':>8s}  "
          f"{'util_mean':>12s}  {'util_std':>10s}  {'max_Δp':>10s}  {'result':>10s}")
    print("-" * 90)

    results_table = []
    combo_num = 0

    for b_cfg in b_configs:
        for cc in cc_values:
            for ms in max_swaps_values:
                combo_num += 1
                b_label = config_label(b_cfg)

                result = run_convergence_test(
                    sim_mode=sim_mode,
                    test_mode="different_start_same_parameter",
                    n_trials=n_trials,
                    network_seed=network_seed,
                    cc=cc,
                    show_comparison=show_comparison,
                    max_swaps=ms,
                    b_config=b_cfg,
                    quiet=True,
                )

                u = result['utilities']
                tag = "UNIQUE" if result['all_match'] else f"{result['n_distinct']} eq."
                if not result['all_converged']:
                    tag = f"!CONV({result['n_converged']}/{n_trials})"

                print(f"{b_label:>12s}  {cc:3d}  {ms:3d}  "
                      f"{result['n_converged']:3d}/{n_trials:<2d}  "
                      f"{result['n_distinct']:5d}/{n_trials:<2d}  "
                      f"{u.mean():12.6f}  {u.std():10.6f}  "
                      f"{result['max_price_diff']:10.2e}  {tag:>10s}")

                results_table.append({
                    'b': b_label,
                    'cc': cc,
                    'max_swaps': ms,
                    **result,
                })

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    n_unique = sum(1 for r in results_table if r['all_match'])
    n_conv = sum(1 for r in results_table if r['all_converged'])
    n_multi = sum(1 for r in results_table if r['all_converged'] and not r['all_match'])

    print(f"Total combos:          {total_combos}")
    print(f"All trials converged:  {n_conv}")
    print(f"Unique equilibrium:    {n_unique}")
    print(f"Multiple equilibria:   {n_multi}")
    print(f"Convergence failures:  {total_combos - n_conv}")

    return results_table


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys

    # Start with configured values
    sim_mode = SIM_MODE
    test_mode = TEST_MODE
    compare_modes = COMPARE_MODES
    compare_swaps = COMPARE_SWAPS
    sweep = False
    cc = CC
    n_trials = N_TRIALS
    network_seed = NETWORK_SEED
    show_comparison = SHOW_COMPARISON
    max_swaps = MAX_SWAPS
    swap_values = None  # default [1, 2]; override with --swap_values=1,2,3
    b_config = None
    a_config = None
    z_config = None

    # Command line overrides
    for arg in sys.argv[1:]:
        if arg == "--aa":
            sim_mode = "aa"
        elif arg == "--async_aa":
            sim_mode = "async_aa"
        elif arg == "--full":
            sim_mode = "full"
        elif arg == "--compare":
            compare_modes = True
        elif arg == "--compare_swaps":
            compare_swaps = True
        elif arg == "--sweep":
            sweep = True
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
        elif arg.startswith("--swap_values="):
            swap_values = [int(v) for v in arg.split("=")[1].split(",")]
        elif arg.startswith("--b_config="):
            b_config = parse_config_arg(arg.split("=", 1)[1])
        elif arg.startswith("--a_config="):
            a_config = parse_config_arg(arg.split("=", 1)[1])
        elif arg.startswith("--z_config="):
            z_config = parse_config_arg(arg.split("=", 1)[1])

    if sweep:
        run_sweep(
            sim_mode=sim_mode,
            n_trials=n_trials,
            network_seed=network_seed,
            show_comparison=show_comparison,
        )
        exit(0)
    elif compare_swaps:
        run_compare_swaps(
            sim_mode=sim_mode,
            test_mode=test_mode,
            n_trials=n_trials,
            network_seed=network_seed,
            cc=cc,
            show_comparison=show_comparison,
            swap_values=swap_values,
        )
        exit(0)
    elif compare_modes:
        result = run_compare_test(
            test_mode=test_mode,
            n_trials=n_trials,
            network_seed=network_seed,
            cc=cc,
            show_comparison=show_comparison,
            max_swaps=max_swaps,
        )
        exit(0 if result else 1)
    else:
        result = run_convergence_test(
            sim_mode=sim_mode,
            test_mode=test_mode,
            n_trials=n_trials,
            network_seed=network_seed,
            cc=cc,
            show_comparison=show_comparison,
            max_swaps=max_swaps,
            b_config=b_config,
            a_config=a_config,
            z_config=z_config,
        )
        exit(0 if result['all_match'] else 1)
