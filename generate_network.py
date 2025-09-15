import random

import igraph
import numpy as np
from itertools import combinations

from functions import draw_random_vector_normal


def initialize_graph_sffa(n: int, c: int):
    exp_in = 1.35 + 1
    exp_out = 1.26 + 1
    i0 = 12

    # fitness vectors
    fitness_in = np.power(np.array(list(range(n))) + 1 + i0 - 1, -1 / (exp_in - 1))
    fitness_out = np.power(np.array(list(range(n))) + 1 + i0 - 1, -1 / (exp_out - 1))

    # create initial graph
    return igraph.Graph.Static_Fitness(n * (c - 1), fitness_out, fitness_in, loops=False, multiple=False)


def initialize_graph_er(n: int, c: int):
    return igraph.Graph.Erdos_Renyi(n=n, m=n*(c-1), loops=False, directed=True)


def initialize_graph(n: int, c: int, topology: str):
    if topology == "SF-FA":
        return add_one_supplier_each_firm(initialize_graph_sffa(n, c))
    elif topology == "ER":
        return add_one_supplier_each_firm(initialize_graph_er(n, c))
    else:
        raise ValueError()


def add_one_supplier_each_firm(initial_graph):
    # add one supplier for each node, such that everybody has a least one supplier
    n = initial_graph.vcount()
    for i in range(n):
        # identify the suppliers
        supplier_id_list = np.empty(n, dtype="object")
        for j in range(n):
            supplier_id_list[j] = initial_graph.neighbors(j, mode="in")
        # identify the potential suppliers
        id_other_firms = list(range(n))
        id_other_firms.remove(i)
        potential_new_supplier_ids_list = list(set(id_other_firms) - set(supplier_id_list[i]))
        # if possible, add a supplier
        if len(potential_new_supplier_ids_list) > 0:
            new_supplier_id = random.sample(potential_new_supplier_ids_list, 1)[0]
            initial_graph.add_edge(new_supplier_id, i)
    return initial_graph


def identify_suppliers(initial_graph):
    # identify the suppliers
    n = initial_graph.vcount()
    nb_suppliers = np.array(initial_graph.degree(list(range(n)), mode="in"))
    supplier_id_list = np.empty(n, dtype="object")
    for i in range(n):
        supplier_id_list[i] = initial_graph.neighbors(i, mode="in")
    return supplier_id_list, nb_suppliers


def create_technology_graph(initial_graph, a: float, b: float, sigma_w: float,
                            nb_suppliers: list[int], nb_extra_suppliers: list[int], supplier_id_list: list):
    n = initial_graph.vcount()
    ## Technology graph
    tech_graph = initial_graph.copy()
    alternate_supplier_id_list = np.empty(n, dtype="object")
    for i in range(n):
        # identify the potential suppliers
        id_other_firms=list(range(n))
        id_other_firms.remove(i)
        potential_alternate_supplier_ids_list = list(set(id_other_firms) - set(supplier_id_list[i]))
        # create the list of alternate suppliers
        if len(potential_alternate_supplier_ids_list) <= nb_extra_suppliers[i]:
            alternate_supplier_id_list[i] = potential_alternate_supplier_ids_list
        else:
            alternate_supplier_id_list[i] = random.sample(potential_alternate_supplier_ids_list, nb_extra_suppliers[i])
        # add the edges in the graph
        for k in alternate_supplier_id_list[i]:
            tech_graph.add_edge(k, i)
        # store it

    tech_graph = add_link_weight(tech_graph, a, b, sigma_w, nb_suppliers)
    # identify the potential suppliers
    nb_pot_suppliers = tech_graph.degree(list(range(n)),mode="in")
    pot_supplier_id_list = np.empty(n, dtype="object")
    for i in range(n):
        pot_supplier_id_list[i] = tech_graph.neighbors(i, mode="in")
    return tech_graph, alternate_supplier_id_list


def add_link_weight(tech_graph, a: float, b: float, sigma_w: float, nb_suppliers: list):
    # create the edge attribute link weight
    eps = 5e-2
    max_wij_tilde = 0
    for v in tech_graph.vs:
        #print("\n"+str(v))
        edge_ids_list = tech_graph.incident(v, mode="in")
        min_val = eps
        max_val = (1-eps)/(b[v.index]*(1-a[v.index]))
        weight_list = draw_random_vector_normal(1/nb_suppliers[v.index], sigma_w, len(edge_ids_list), min_val=min_val, max_val=max_val).tolist()
        igraph.EdgeSeq(tech_graph, edge_ids_list)["weight"] = weight_list
        #print(weight_list)
        max_wij_tilde = max(max_wij_tilde, max([item*b[v.index]*(1-a[v.index]) for item in weight_list]))
        #for edge_id in edge_ids_list:
        #  drawn_value = 1/nb_suppliers[v.index]*(1+np.random.normal(0, sigma_w))
        #min_val = 0.1
        #max_val = 100
        #   drawn_value = max(min(drawn_value, max_val), min_val)
        #    weight_list.append(drawn_value)
        #weight_list = 1/nb_suppliers[v.index]*(1+np.random.normal(0, sigma_w))
        #weight_list = max(0.1, weight_list)
    # print(("max_wij_tilde: ", max_wij_tilde))
    return tech_graph


def get_tech_matrix(tech_graph):
    return np.array(tech_graph.get_adjacency(attribute="weight", default=0).data)


def get_initial_matrices(initial_graph, tech_graph):
    M0 = np.array(initial_graph.get_adjacency(attribute=None).data)
    Wbar = get_tech_matrix(tech_graph)
    W0 = M0 * Wbar
    return M0, W0, Wbar


def get_AiSi_productivities(supplier_id_list: list, alternate_supplier_id_list: list, spread: float):
    nb_suppliers_per_firm = [len(s) for s in supplier_id_list]
    all_potential_suppliers_per_firm = [sorted(supplier_id_list[i] + alternate_supplier_id_list[i])
                                        for i in range(len(supplier_id_list))]
    all_combinations_of_suppliers = [list(combinations(all_suppliers, c_i)) for c_i, all_suppliers
              in zip(nb_suppliers_per_firm, all_potential_suppliers_per_firm)]
    AiSi = [{combi: random.uniform(1 - spread, 1 + spread) for combi in all_combi}
            for all_combi in all_combinations_of_suppliers]
    return AiSi

def compute_adjusted_z(z: np.array, AiSi: list, supplier_id_list: list):
    current_AiSi = [AiSi_one_firm[tuple(sorted(supplier_id_list[i]))] for i, AiSi_one_firm in enumerate(AiSi)]
    return z * np.array(current_AiSi)


def regenerate_network_from_cached_parameters(a: np.array, b: np.array, z: np.array, 
                                              Wbar: np.array, AiSi: list):
    """
    Generate a new initial network by randomly selecting supplier combinations 
    from cached AiSi parameters.
    
    This function creates a new network configuration by randomly selecting,
    for each firm, one of the possible supplier combinations stored in AiSi.
    The firm-level parameters (a, z, b) and technology matrix (Wbar) remain unchanged.
    
    Parameters:
    -----------
    a : np.array
        Labor share parameters for each firm (unchanged)
    b : np.array  
        Returns to scale parameters for each firm (unchanged)
    z : np.array
        Productivity parameters for each firm (unchanged)
    Wbar : np.array
        Technology matrix with link weights (unchanged)
    AiSi : list
        List of dictionaries, one per firm, where each dictionary maps 
        supplier combinations (tuples) to productivity multipliers
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'supplier_id_list': New list of selected suppliers for each firm
        - 'alternate_supplier_id_list': Remaining suppliers for each firm
        - 'initial_graph': New igraph network based on selected suppliers
        - 'M0': New initial adjacency matrix
        - 'W0': New initial weighted matrix  
    """
    nb_firms = len(a)
    
    # Step 1: Randomly select supplier combinations for each firm
    new_supplier_id_list = []
    for firm_id in range(nb_firms):
        # Get all possible supplier combinations for this firm
        possible_combinations = list(AiSi[firm_id].keys())
        
        # Randomly select one combination
        selected_combination = random.choice(possible_combinations)
        
        # Convert tuple to list and store
        new_supplier_id_list.append(list(selected_combination))
    
    # Step 2: Build alternate supplier lists
    # For each firm, alternates are all other possible suppliers not currently selected
    new_alternate_supplier_id_list = []
    for firm_id in range(nb_firms):
        # Get all possible suppliers from all combinations in AiSi
        all_possible_suppliers = set()
        for combination in AiSi[firm_id].keys():
            all_possible_suppliers.update(combination)
        
        # Remove current suppliers to get alternates
        current_suppliers = set(new_supplier_id_list[firm_id])
        alternates = list(all_possible_suppliers - current_suppliers)
        new_alternate_supplier_id_list.append(alternates)
    
    # Step 3: Create new igraph network from selected suppliers
    import igraph
    new_initial_graph = igraph.Graph(directed=True)
    new_initial_graph.add_vertices(nb_firms)
    
    # Add edges based on selected supplier relationships
    edges_to_add = []
    for customer_id in range(nb_firms):
        for supplier_id in new_supplier_id_list[customer_id]:
            edges_to_add.append((supplier_id, customer_id))
    
    new_initial_graph.add_edges(edges_to_add)
    
    # Step 4: Create new adjacency matrices
    M0_new = np.array(new_initial_graph.get_adjacency(attribute=None).data)
    W0_new = M0_new * Wbar
    
    return {
        'supplier_id_list': new_supplier_id_list,
        'alternate_supplier_id_list': new_alternate_supplier_id_list,
        'initial_graph': new_initial_graph,
        'M0': M0_new,
        'W0': W0_new
    }
    