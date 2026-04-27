"""Network generation (SF-FA topology) and helpers.

Public API:
    initialize_sffa_adjacency        -- raw SF-FA adjacency
    add_one_supplier_each_firm       -- guarantee at least one supplier per firm
    identify_suppliers_from_matrix   -- column-wise supplier extraction
    create_technology_matrix         -- weights and alternates
    get_AiSi_productivities          -- supplier-combination-specific multipliers
    build_W_from_suppliers           -- assemble W from a supplier-set list
    generate_base_network            -- end-to-end network state for a fresh economy
    generate_random_initial_network  -- random initial supplier pick on a fixed Wbar
"""

import random
from itertools import combinations

import numpy as np

from .parameters import draw_random_vector_normal


def initialize_sffa_adjacency(n: int, c: int) -> np.ndarray:
    """Create initial adjacency M with M[supplier, customer] = 1, SF-FA topology."""
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
    """Extract supplier lists and degrees from an adjacency matrix."""
    n = M.shape[0]
    supplier_id_list = [list(np.where(M[:, i] == 1)[0]) for i in range(n)]
    nb_suppliers = np.array([len(s) for s in supplier_id_list])
    return supplier_id_list, nb_suppliers


def create_technology_matrix(M: np.ndarray, nb_suppliers: np.ndarray,
                             nb_extra_suppliers: np.ndarray, supplier_id_list: list,
                             a: np.ndarray = None, b: np.ndarray = None,
                             sigma_w: float = 0.0):
    """Build the technology matrix Wbar (link weights) and alternate-supplier list."""
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

        mean_weight = 1.0 / nb_suppliers[i] if nb_suppliers[i] > 0 else 1.0

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
    """Random productivity multiplier for each (firm, supplier-combination) pair."""
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


def build_W_from_suppliers(supplier_set: list, Wbar: np.ndarray) -> np.ndarray:
    """Build the active W matrix from a per-firm supplier-set list."""
    n = Wbar.shape[0]
    W = np.zeros((n, n), dtype=Wbar.dtype)
    for buyer, suppliers in enumerate(supplier_set):
        if suppliers:
            W[list(suppliers), buyer] = Wbar[list(suppliers), buyer]
    return W


# -----------------------------------------------------------------------------
# Higher-level factory: one call to build a complete network state
# -----------------------------------------------------------------------------

def generate_base_network(n, c, cc, aisi_spread, seed=None,
                          a=None, b=None, sigma_w=0.0):
    """Build a full network state (M0, W0, Wbar, supplier lists, AiSi).

    Re-seeds Python and numpy RNGs from `seed` if provided.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    M0 = add_one_supplier_each_firm(initialize_sffa_adjacency(n, c))
    supplier_id_list, nb_suppliers = identify_suppliers_from_matrix(M0)
    Wbar, alternate_supplier_id_list = create_technology_matrix(
        M0, nb_suppliers, np.full(n, cc), supplier_id_list,
        a=a, b=b, sigma_w=sigma_w,
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
    """Pick a random initial supplier set per firm from the AiSi-supported combinations.

    Wbar and AiSi are kept fixed; only the active supplier set varies.
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
