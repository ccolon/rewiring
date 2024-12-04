import random

import igraph
import numpy as np

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


def initialize_graph(n: int, topology: str):
    if topology == "SF-FA":
        return add_one_supplier_each_firm(initialize_graph_sffa(n, 4))
    elif topology == "ER":
        return add_one_supplier_each_firm(initialize_graph_er(n, 4))
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

# Mbar = np.array(tech_graph.get_adjacency(attribute=None).data)
#print(Mbar)


    