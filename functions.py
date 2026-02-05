import numpy as np
from scipy.sparse.linalg import eigs

from parameters import EPSILON


def get_alpha(a, W):
    return a + (1 - a) * np.sum(W, axis=0)


def compute_cost_gap(a, b, z, W, n, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm):
    '''Compute a network-level metric. It is the mean of all so-called firm-level penalty.
    A firm's penalty is the difference between the maximum profit reachable with perfect anticipation and the current profit
    '''
    all_indiv_score = [
        evalute_cost_penalty(firm_id, a, b, z, W, n, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
        for firm_id in range(n)]
    return np.mean(all_indiv_score)


def calculate_utility(eq):
    prices = eq['P']
    prices = prices[prices > 0]
    return -np.sum(prices)



def compute_equilibrium(a, b, z, W, n, shot_firm=None):  # with solve instead of inv
    if isinstance(shot_firm, int):
        a = np.delete(a, shot_firm)  # Creates a new array for a
        z = np.delete(z, shot_firm)  # Creates a new array for a
        b = np.delete(b, shot_firm)  # Creates a new array for a
        W = np.delete(W, shot_firm, axis=0)  # Creates a new array for W
        W = np.delete(W, shot_firm, axis=1)
        n = n - 1
    alpha = get_alpha(a, W)
    M = (1 / alpha) * (a / n + (1 - a)[np.newaxis, :] * W)
    if M.shape[0] > 10:
        eigenvalues, eigenvectors = eigs(M, k=1, which='LM')  # 'LM' for Largest Magnitude
        eigenvalue = eigenvalues[0].real
        if abs(1.0 - eigenvalue) > EPSILON:
            print(M)
            raise ValueError("Eigenvalue is not 1")
        v_unnormalized = eigenvectors[:, 0].real
    else:
        eigenvalues, eigenvectors = np.linalg.eig(M)
        index = np.isclose(eigenvalues, 1)
        if ~index.any():
            raise ValueError('No eigenvalue 1')
        eigenvectors_1 = eigenvectors[:, index]
        v_unnormalized = np.transpose(eigenvectors_1)
    # check no imaginary part
    if np.sum(np.imag(v_unnormalized)) > EPSILON:
        raise ValueError("Imaginary part in the eigenvector")
    v_unnormalized = np.real(v_unnormalized)
    # check all same sign
    if (v_unnormalized > 0).any() and (v_unnormalized < 0).any():
        raise ValueError("Negative sales")
    v_unnormalized = abs(v_unnormalized).flatten()
    if np.sum(v_unnormalized) < EPSILON:
        raise ValueError("All sales are null")

    kappa = n / np.sum(v_unnormalized * a / alpha)
    v = kappa * v_unnormalized

    # Get p, by passing prod function to the log and solve Ap = b
    b_vector = (-np.log(z) + b * alpha * np.log(alpha) + (1 - b * alpha) * np.log(v))[:, np.newaxis]
    A_matrix = np.eye(n) - (b * (1 - a))[:, np.newaxis] * W.T

    log_p = np.linalg.solve(A_matrix, b_vector)
    p = np.exp(log_p)

    # Get x
    v = v.flatten()
    p = p.flatten()
    x = v / p

    check = False
    if check:
        l = v * a / alpha
        # G = np.transpose(v * (1 - a) / alpha * W.T / np.transpose(p))
        # G = v * (1 - a) * W / alpha / p[:, np.newaxis]
        G = (W * (v * (1 - a) / alpha)[None, :]) / p[:, None]
        L = np.sum(l)
        B = L
        print("Quantity of labor", L, "Budget", B)
        final_demand = B / (n * p)
        intermediary_demand = np.sum(G, axis=1)
        supply = x
        print("supply", supply)
        print("total demand", intermediary_demand + final_demand)
        print("intermediary_demand", intermediary_demand)
        print("final_demand", final_demand)
        print("dif", supply - (intermediary_demand + final_demand) > 1e-10)

        # Check for CRS case
        p1_calculated = np.prod(np.power(p, (1-a[1]) * W[:,1])) / z[1]
        print("b1", b[1], "alpha1", alpha[1], "sumWcol1", W[:, 1].sum())
        print(f"p1 {p[1]}", f"p1_calculated {p1_calculated}")

        log_p1_calc = (
                -np.log(z[1])
                + b[1] * alpha[1] * np.log(alpha[1])
                + (1 - b[1] * alpha[1]) * np.log(v[1])
                + np.sum(b[1] * (1 - a[1]) * W[:, 1] * np.log(p))
        )
        p1_calculated = np.exp(log_p1_calc)
        print(p[1], p1_calculated, p[1] - p1_calculated)

        p1_col = np.exp(-np.log(z[1]) + (1 - a[1]) * np.sum(W[:, 1] * np.log(p)))
        p1_row = np.exp(-np.log(z[1]) + (1 - a[1]) * np.sum(W[1, :] * np.log(p)))

        print("p[1]    =", p[1])
        print("col chk =", p1_col)  # uses W[:,1]
        print("row chk =", p1_row)  # uses W[1,:]

        exit()

    productions = x.flatten()
    prices = p.flatten()
    if isinstance(shot_firm, int):
        productions = np.insert(productions, shot_firm, 0.0)
        prices = np.insert(prices, shot_firm, 0.0)

    return {"X": productions, "P": prices}


def compute_cost(firm_id, a, b, W, X, P, h):
    labor_need = a[firm_id] * b[firm_id] * P[firm_id] * X[firm_id] / h
    good_needs = (1 - a[firm_id]) * b[firm_id] * P[firm_id] * X[firm_id] * np.transpose(W[:, firm_id]) / P
    cost = h * labor_need + np.sum(P * good_needs)
    return cost


def evaluate_best_alternative_cost(firm_id, a, b, z, W, n, Wbar, supplier_id_list, alternate_supplier_id_list,
                                   shot_firm):
    """Compute the best profit reachable by a switch
    Full anticipation version: the firm updates the W based on the switch and computes the new network-wide equilibrium
    """
    min_cost = 9999
    for id_visited_supplier in alternate_supplier_id_list[firm_id]:
        W[id_visited_supplier, firm_id] = Wbar[id_visited_supplier, firm_id]
        for id_replaced_supplier in supplier_id_list[firm_id]:
            #print('test', firm_id, id_replaced_supplier, id_visited_supplier)
            W[id_replaced_supplier, firm_id] = 0  # on enleve ce lien dans le W
            new_eq = compute_equilibrium(a, b, z, W, n, shot_firm)
            new_cost = new_eq['P'][firm_id]  # compute_cost(firm_id, a, b, W, new_eq['X'], new_eq['P'], new_eq['h'])
            if new_cost < min_cost:
                min_cost = new_cost
            W[id_replaced_supplier, firm_id] = Wbar[
                id_replaced_supplier, firm_id]  #apres le test d'un supplier, on remet le lien dans W
        W[id_visited_supplier, firm_id] = 0  # a la fin du test, on remet W comme avant
    return min_cost


def evalute_cost_penalty(firm_id, a, b, z, W, n, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm):
    """Compute the difference between the current profit and the max profit reachable by a switch
    """
    eq = compute_equilibrium(a, b, z, W, n, shot_firm)
    current_cost = eq['P'][firm_id]  # compute_cost(firm_id, a, b, W, eq['X'], eq['P'], eq['h'])
    min_alternative_cost = evaluate_best_alternative_cost(firm_id, a, b, z, W, n, Wbar, supplier_id_list,
                                                          alternate_supplier_id_list, shot_firm)
    dif = current_cost - min_alternative_cost
    if dif <= 0:
        # print('Firm '+str(firm_id)+': at the best profit')
        return 0
    else:
        # print('Firm '+str(firm_id)+': not at the best profit')
        return abs(dif) / current_cost


def compute_partial_equilibrium_and_cost(a, b, z, W, firms_within_tiers, id_rewiring_firm,
                                         shot_firm):
    """Compute equilibrium for selected firms only
    Suppose that the firm doing this calculation knows:
    - prices, production, wage from last time step
    - all parameters of firms_within_tiers
    - the amount of work hired and intermediary demand from last time step of firms_within_tiers
    - the matrix W within firms_within_tiers
    """
    # reduce the system
    W_reduced_unnormalized = W[np.ix_(firms_within_tiers, firms_within_tiers)]
    col_sums = W_reduced_unnormalized.sum(axis=0)
    nonzero_mask = col_sums != 0.0
    W_reduced = W_reduced_unnormalized.copy()
    # W_reduced[:, nonzero_mask] = W_reduced_unnormalized[:, nonzero_mask] / col_sums[nonzero_mask]
    a_reduced = a[firms_within_tiers]
    # a_reduced[~nonzero_mask] = 1
    b_reduced = b[firms_within_tiers]
    z_reduced = z[firms_within_tiers]
    n_reduced = len(firms_within_tiers)

    # # First equation: good market balance
    # if shot_firm == None:  # we do not use n_reduced here because:
    #     # 1. the id of the shot_firm correspoonds to the non-reduced system
    #     # 2. the wealth corresponds to the non-reduced system
    #     hh_demand = (wealth / n) * np.ones([n, 1])
    # else:
    #     hh_demand = (wealth / (n - 1)) * np.ones([n, 1])
    #     hh_demand[shot_firm] = 0

    shot_firm_reduced = None
    if shot_firm in firms_within_tiers:
        shot_firm_reduced = firms_within_tiers.index(shot_firm)
    partial_eq = compute_equilibrium(a_reduced, b_reduced, z_reduced, W_reduced, n_reduced, shot_firm_reduced)
    partial_eq["firms_within_tiers"] = firms_within_tiers

    firm_id_reduced = firms_within_tiers.index(id_rewiring_firm)
    mean_cost = partial_eq['P'][firm_id_reduced]

    return partial_eq, mean_cost


def compute_distance_btw_ntw(g1, g2):
    # Compute distance btw ntw. First version, using adjacency matrix
    M1 = np.array(g1.get_adjacency().data)
    M2 = np.array(g2.get_adjacency().data)
    return np.sum(np.sum(M1 * M2)) / np.sqrt(g1.ecount() * g2.ecount())


def compute_distance_btw_ntw2(g1, g2):
    # Compute distance btw ntw. Second version, using edgelist
    EL1 = np.array(g1.get_edgelist())
    EL2 = np.array(g2.get_edgelist())
    nb_common_rows_of2d_arrays(EL1, EL2)
    return nb_common_rows_of2d_arrays(EL1, EL2) / np.sqrt(g1.ecount() * g2.ecount())


# Compute the number of common rows in 2d arrays
def nb_common_rows_of2d_arrays(A, B):
    nrows, ncols = A.shape
    dtype = {'names': ['f{}'.format(i) for i in range(ncols)],
             'formats': ncols * [A.dtype]}
    C = np.intersect1d(A.view(dtype), B.view(dtype))
    C = C.view(A.dtype).reshape(-1, ncols)
    nrows, ncols = C.shape
    return nrows


def draw_random_vector_normal(mean, sd, n, min_val=None, max_val=None):
    vec = np.random.normal(mean, sd, n)
    if min_val or max_val:
        for k in range(len(vec)):
            if min_val:
                if vec[k] < min_val:
                    vec[k] = min_val
            if max_val:
                if vec[k] > max_val:
                    vec[k] = max_val
    return vec


def draw_random_vector_lognormal(mean, sd, n, min_val=None, max_val=None, integer=False):
    if mean == 0:
        return [0 for x in range(n)]

    else:
        mean_lognorm = np.log(mean ** 2 / (mean ** 2 + sd ** 2) ** (1 / 2))
        sd_lognorm = (np.log(1 + sd ** 2 / mean ** 2)) ** (1 / 2)
        vec = np.random.lognormal(mean_lognorm, sd_lognorm, n)
        if min_val or max_val:
            for k in range(len(vec)):
                if min_val:
                    if vec[k] < min_val:
                        vec[k] = min_val
                if max_val:
                    if vec[k] > max_val:
                        vec[k] = max_val
        if integer:
            vec = [int(round(x)) for x in list(vec)]
        return vec


def identify_firms_within_tier(id_firm, g, tier):
    neighboors = g.neighborhood(vertices=id_firm, order=tier, mode='all')
    neighboors.sort()
    return neighboors


def build_W_star_from_best_sets(
    aa_best_supplier_set,
    Wbar: np.ndarray,
    n: int | None = None,
) -> np.ndarray:
    """
    Build W_star (supplier->buyer IO matrix) from aa_best_supplier_set and Wbar.

    Convention: W[supplier, buyer] = w_{supplier,buyer}.
    aa_best_supplier_set[buyer] is an iterable of supplier ids chosen for that buyer.
    Wbar contains the predetermined coefficients; non-chosen links are set to 0.

    Note: this preserves the Wbar column-sum condition if Wbar already satisfies it
    on the chosen suppliers (as in your setup).
    """
    if n is None:
        n = Wbar.shape[0]

    W_star = np.zeros((n, n), dtype=Wbar.dtype)

    for buyer, suppliers in enumerate(aa_best_supplier_set):
        # allow list/tuple/set/np.array
        if suppliers is None:
            continue
        suppliers = list(suppliers)
        if len(suppliers) == 0:
            continue

        W_star[suppliers, buyer] = Wbar[suppliers, buyer]

    return W_star