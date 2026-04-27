"""Rewiring simulation drivers.

Public API:
    run_unified_simulation     -- canonical driver supporting all anticipation modes
                                  ('aa', 'full', 'limited', 'naive_limited')
    run_async_aa_simulation    -- legacy AA driver (Gauss-Seidel async, kept because
                                  it's not strictly subsumed by the unified driver)

Both drivers are asynchronous: within a round, firms decide one at a time in a
random permutation order; an accepted swap is applied immediately and the GE is
recomputed before the next firm decides. Convergence: zero swaps in a full round.
"""

import random
from itertools import combinations

import numpy as np

from .equilibrium import (
    calculate_utility,
    compute_adjusted_z,
    compute_equilibrium_crs,
    compute_equilibrium_full,
)
from .networks import build_W_from_suppliers
from .parameters import EPSILON
from .partial_eq import (
    compute_partial_equilibrium_cost,
    compute_partial_equilibrium_cost_naive,
    identify_firms_within_tier_np,
)


# Default round budget; callers should normally pass an explicit `nb_rounds`.
NB_ROUNDS = 20


def _edges_from_suppliers(supplier_id_list):
    """Return (K, 2) int32 array of (supplier, buyer) pairs, lexicographically sorted."""
    pairs = [(int(s), int(buyer))
             for buyer, suppliers in enumerate(supplier_id_list)
             for s in suppliers]
    pairs.sort()
    return np.asarray(pairs, dtype=np.int32)


def _state_signature(supplier_id_list):
    """Hashable canonical signature of the supplier-set configuration.

    Two states with the same active suppliers per firm produce equal signatures
    regardless of the order in which the suppliers were appended to each list.
    """
    return tuple(tuple(sorted(int(x) for x in s)) for s in supplier_id_list)


# =============================================================================
# UNIFIED ASYNC SIMULATION (aa / full / limited / naive_limited)
# =============================================================================

def run_unified_simulation(network_state, a, b, z, mode="aa", seed=None,
                           max_swaps=1, nb_rounds=None, tier=None, trace=False):
    """Unified asynchronous rewiring simulation.

    Common structure for all anticipation modes:
    - Round = one permutation pass over firms.
    - Each firm evaluates all combinations up to max_swaps against its current
      supplier set, using a mode-specific cost function.
    - If the best candidate strictly reduces the (mode-specific) cost, the swap
      is applied IMMEDIATELY, the true GE is recomputed, and the next firm in
      the permutation sees the updated state.
    - Convergence: no swaps during a full round.

    Mode differs only in the candidate-cost function:
    - 'aa': closed-form ratchet -- prod(P^((1-a_i)*W_col')) / test_z
            where P is the current equilibrium price vector.
    - 'full': solve compute_equilibrium_full on the hypothetical W (current W
              with firm i's column replaced); take the i-th price.
    - 'limited': boundary-conditioned partial equilibrium on the tier
                 neighborhood of the hypothetical graph.
    - 'naive_limited': legacy island-only partial equilibrium.

    Args:
        network_state: dict from generate_base_network.
        a, b, z: economic parameter arrays of length n.
        mode: 'aa', 'full', 'limited', or 'naive_limited'.
        seed: random seed for permutation order.
        max_swaps: max simultaneous supplier swaps.
        nb_rounds: number of rounds (defaults to module-level NB_ROUNDS).
        tier: int or np.ndarray of length n. Required for 'limited' / 'naive_limited'.
        trace: if True, attach a `trace` dict to the result with per-round scalars,
               edge snapshots, per-step prices, and rewire events.
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

    if mode in ("limited", "naive_limited"):
        tier_arr = np.full(n, int(tier)) if np.isscalar(tier) else np.asarray(tier, dtype=int)

    adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
    eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
    initial_utility = calculate_utility(eq)

    one_minus_a = 1 - a

    def candidate_cost(id_firm, candidate_set):
        """Cost of id_firm if its supplier set were `candidate_set`, under mode."""
        W_col = np.zeros(n)
        for s in candidate_set:
            W_col[s] = Wbar[s, id_firm]
        test_z = AiSi[id_firm][tuple(sorted(int(s) for s in candidate_set))]

        if mode == "aa":
            return float(np.prod(np.power(eq['P'], one_minus_a[id_firm] * W_col)) / test_z)

        # For full / limited variants: build W_test by replacing firm i's column
        W_test = W.copy()
        W_test[:, id_firm] = W_col

        tmp_supplier_list = [list(s) for s in supplier_id_list]
        tmp_supplier_list[id_firm] = sorted(int(s) for s in candidate_set)
        tmp_adjusted_z = compute_adjusted_z(AiSi, tmp_supplier_list, z)

        if mode == "full":
            new_eq = compute_equilibrium_full(a, b, tmp_adjusted_z, W_test, n)
            return float(new_eq['P'][id_firm])

        # Tier neighborhood on the hypothetical graph (used by both limited variants)
        M_test = (W_test != 0).astype(np.int8)
        firms_within_tiers = identify_firms_within_tier_np(M_test, id_firm, tier_arr[id_firm])

        if mode == "limited":
            return compute_partial_equilibrium_cost(
                a, b, tmp_adjusted_z, W, W_test, eq,
                firms_within_tiers, id_firm, n,
            )

        # mode == "naive_limited"
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

    # Cycle detection: track per-round canonical state signatures so we can
    # recognise period-2 limit cycles (state_t == state_{t-2} and
    # state_{t-1} == state_{t-3}). cycle_period:
    #   1    -> strict convergence (no rewires in a full round)
    #   2    -> period-2 limit cycle
    #   None -> hit nb_rounds budget without either
    state_history = [_state_signature(supplier_id_list)]
    cycle_period = None

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

        # Strict convergence: no rewires this round.
        if rewirings_this_round == 0:
            cycle_period = 1
            if trace:
                converged_at = r
            break

        # Period-2 limit cycle: current state matches 2 rounds ago, and the
        # previous round matched 3 rounds ago. Two matching pairs guard against
        # a coincidental single-round equality.
        state_history.append(_state_signature(supplier_id_list))
        if (len(state_history) >= 4
                and state_history[-1] == state_history[-3]
                and state_history[-2] == state_history[-4]):
            cycle_period = 2
            if trace:
                converged_at = r
            break

    if trace:
        # Anchor the last observation at the final step so the price trajectory
        # extends to t_final even when the run has flat tails.
        if not trace_price_steps or trace_price_steps[-1] != t:
            trace_prices.append(eq['P'].copy())
            trace_price_steps.append(t)

    result = {
        # `converged` keeps its strict meaning (period-1 only) for backward
        # compat with existing post-processors. Use `cycle_period` to also
        # distinguish "limit cycle" from "budget exhausted".
        'converged': cycle_period == 1,
        'cycle_period': cycle_period,
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
# ASYNC AA MODE SIMULATION
# =============================================================================

def run_async_aa_simulation(network_state, a, seed=None, max_swaps=1, nb_rounds=None,
                            b=None, z=None):
    """AA simulation with asynchronous (Gauss-Seidel) updates.

    Cost formula (same as the AA closed form):
        cost = prod(P_ref^((1-a)*W_col)) / z_i

    The swap is applied immediately after each firm decides, and the true
    equilibrium is recomputed right away. The next firm in the permutation
    uses the updated P_ref. Convergence: no rewirings during a full round.

    CRS path (b is None): uses compute_equilibrium_crs and a scalar `a`.
    Heterogeneous path (b provided): uses compute_equilibrium_full with arrays.
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

    if crs_mode:
        adjusted_z = compute_adjusted_z(AiSi, supplier_id_list)
        eq = compute_equilibrium_crs(a, adjusted_z, W, n)
        initial_utility = -eq['P'].sum()
    else:
        adjusted_z = compute_adjusted_z(AiSi, supplier_id_list, z)
        eq = compute_equilibrium_full(a, b, adjusted_z, W, n)
        initial_utility = calculate_utility(eq)

    P_ref = eq['P'].copy()
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
