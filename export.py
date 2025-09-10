import os
import re
from datetime import datetime

import igraph
import numpy as np
import pandas as pd


def create_general_output_folder(exp_type, exp_name):
    general_output_folder = os.path.join("output", exp_type + '_' + exp_name)
    necessary_folder = ["tmp", "output", general_output_folder]
    for folder in necessary_folder:
        if not os.path.exists(folder):
            os.makedirs(folder)
    return general_output_folder


def create_export_folder(exp_name):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    output_folder = os.path.join('output', current_time + '_' + exp_name)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    return output_folder


def initialize_ts_file(output_folder, export_firm_profits, nb_firms=0):
    file = open(os.path.join(output_folder, "results.txt"), "w")
    if export_firm_profits:
        line = "t rewiring_ts utility_ts " + ' ' \
               + ' '.join('x' + str(i) for i in range(nb_firms)) + " "\
               + ' '.join('p' + str(i) for i in range(nb_firms)) + '\n'
    else:
        line = "t rewiring_ts utility_ts score\n"
    file.write(line)
    return file


def append_ts_file(file, t: int, rewiring_ts, utility_ts, eq, export_prices_productions, score=None):
    if export_prices_productions:
        line = str(t) + ' ' + str(rewiring_ts) + ' ' + str(utility_ts) + ' ' \
               + " ".join(map(str, eq['X'])) + ' ' + " ".join(map(str, eq['P'])) + '\n'
    elif score:
        line = str(t) + ' ' + str(rewiring_ts) + ' ' + str(utility_ts) + ' ' + str(score) + "\n"
    else:
        line = str(t) + ' ' + str(rewiring_ts) + ' ' + str(utility_ts) + "\n"
    file.write(line)


def initialize_rewiring_file(output_folder):
    file_who_rewire = open(os.path.join(output_folder, "who_rewire.txt"), "w")
    file_who_rewire.write("round ts who_rewire\n")
    return file_who_rewire


def export_parameters(output_folder, nb_round, topology, nb_firms, c, cc, sigma_a, sigma_b, sigma_z, sigma_w,
                      a, b, z, nb_suppliers, nb_extra_suppliers, initial_graph, tech_graph, save_networks):
    global_param_list = pd.DataFrame(
        data={"NbRound": [nb_round], "topology": [topology], "n": [nb_firms], "c": [c], "cc": [cc],
              "sigma_w": [sigma_w],
              "sigma_z": [sigma_z], "sigma_b": [sigma_b], "sigma_a": [sigma_a]})
    global_param_list.to_csv(os.path.join(output_folder, 'global_param_list.txt'), sep=" ", index=False)
    firm_param_list = pd.DataFrame(
        data={"a": a, "b": b, "z": z, "nb_suppliers": nb_suppliers, "nb_extra_suppliers": nb_extra_suppliers})
    firm_param_list.to_csv(os.path.join(output_folder, 'firm_param_list.txt'), sep=" ", index=False)
    np.array(initial_graph.get_edgelist()).tofile(os.path.join(output_folder, 'M_0.txt'), sep=" ")

    # Output tech network
    if save_networks:
        # np.array(tech_graph.get_edgelist()).tofile(output_folder + 'Mbar_.txt', sep = " ")
        np.array(igraph.EdgeSeq(tech_graph)["weight"]).tofile(os.path.join(output_folder, 'Wbar_edgelist.txt'), sep=" ")


def compute_gini_coef(dist):
    if sum(dist) == 0:
        return 0
    else:
        n = len(dist)
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += abs(dist[i] - dist[j])
        denominator = 2 * n ** 2 * sum(dist) / n
        return numerator / denominator


def edgevector_from_file(filename):
    with open(filename, 'r') as file:
        edgevector = file.read()
    edgevector = str(edgevector).split(' ')
    return [edgevector[i] + ' ' + edgevector[i + 1] for i in range(len(edgevector)) if i % 2 == 0]


def get_all_edgevectors(folder):
    all_M_filenames = os.listdir(folder)  # , pattern = "M_[0-9*]", full.names = T)
    all_M_filenames = [item for item in all_M_filenames if re.match(r"M_(\d*).txt", str(item))]
    all_M_filenames = {int(re.match(r"M_(\d*).txt", str(item)).group(1)): item for item in all_M_filenames}
    edgevectors = {ts: edgevector_from_file(os.path.join(folder, filename)) for ts, filename in
                   list(all_M_filenames.items())}
    return edgevectors



def compute_scaled_distance(edgevector1, edgevector2):
    return 1.0 - float(len(set(edgevector1) & set(edgevector2))) / len(edgevector1)


def compute_distance_matrix(folder):
    edgevectors = get_all_edgevectors(folder)
    nb_networks = len(edgevectors)
    print(("Computing distance between", nb_networks, "networks."))
    res = np.zeros((nb_networks, nb_networks), dtype=int)
    for i in range(nb_networks):
        for j in range(i + 1, nb_networks):
            res[i, j] = compute_scaled_distance(list(edgevectors.values())[i], list(edgevectors.values())[j])
    res = pd.DataFrame(res, index=list(edgevectors.keys()), columns=list(edgevectors.keys()))
    return res


def export_summary_file(output_folder, eq_reached, t, r, total_time):
    file = open(os.path.join(output_folder, "eq_and_time.txt"), "w")
    file.write("eq_reached tfinal nb_rounds total_time\n")
    file.write(str(eq_reached) + ' ' + str(t) + ' ' + str(r) + ' ' + str(total_time) + "\n")
    file.close()


def export_final_matrix(output_folder, graph):
    np.array(graph.get_edgelist()).tofile(os.path.join(output_folder, 'M_final.txt'), sep = " ")

