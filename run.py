import os
import pickle
import random
import subprocess
import sys

import igraph
import numpy as np
import pandas as pd

from export import create_export_folder, initialize_ts_file, initialize_rewiring_file, export_parameters, \
    append_ts_file, export_summary_file, export_final_matrix, create_general_output_folder
from functions import draw_random_vector_normal, compute_cost_gap, identify_firms_within_tier, \
    draw_random_vector_lognormal, compute_equilibrium, compute_partial_equilibrium_and_cost, calculate_utility
from generate_network import initialize_graph, create_technology_graph, identify_suppliers, get_tech_matrix, \
    get_initial_matrices
from parameters import *

# Model parameters
if len(sys.argv) > 1:
    print('ok')
    sigma_w = float(sys.argv[5])
    sigma_z = float(sys.argv[6])
    sigma_b = float(sys.argv[7])
    sigma_a = float(sys.argv[8])
    rewiring_test = 'try_all'

    # Whether to reuse an existing graph
    if sys.argv[9] == "inputed":
        inputed_network = True
        novelty_suffix = ""
    else:
        inputed_network = False
        novelty_suffix = '_NEWNET'

    # Do runs
    nb_rounds = int(sys.argv[2])
    nb_firms = int(sys.argv[3])
    c = 4
    cc = int(sys.argv[4])

#================================================================================
# Retrieve parameter name
general_output_folder = create_general_output_folder(exp_type, exp_name)
# Start counting time
starting_time = datetime.now()

#================================================================================
# Parameters that need n
nb_extra_suppliers = np.full(nb_firms, cc)

if inputed_network:
    cache_initial_network = False
    cache_firm_parameters = False
    a = pickle.load(open('tmp/a', 'rb'))
    b = pickle.load(open('tmp/b', 'rb'))
    z = pickle.load(open('tmp/z', 'rb'))
else:
    cache_initial_network = True
    cache_firm_parameters = True
    # Economic parameters: global return to scale b
    # b
    #b = np.full(n, 0.9)
    eps = 5e-2
    min_b = eps
    max_b = 1 - eps
    mean_b = 0.9
    b = draw_random_vector_normal(mean_b, sigma_b, nb_firms, min_b, max_b)
    #nb_different = 30
    #b[0:nb_different] = b[0:nb_different]+1
    print("b: min " + str(min(b)) + ' max ' + str(max(b)))

    # Economic parameters: labor share a
    eps = 5e-2
    min_a = eps
    a = np.array(
        [draw_random_vector_normal(0.5, sigma_a, nb_firms, min_a, min((1 - eps) / item, 1 - eps))[0] for item in
         list(b)])
    print("a: min " + str(min(a)) + ' max ' + str(max(a)))
    #a[2] = 0.97

    # Economic parameters: Productivity z
    min_z = 1e-1
    z = draw_random_vector_normal(1, sigma_z, nb_firms, min_val=min_z)
    print("z: min " + str(min(z)) + ' max ' + str(max(z)))

if cache_firm_parameters:
    pickle.dump(a, open('tmp/a', 'wb'))
    pickle.dump(b, open('tmp/b', 'wb'))
    pickle.dump(z, open('tmp/z', 'wb'))

#================================================================================
# Create tech network Wbar and initial input-output network W0
## Option 1: Load existing network. needs g0, techgraph, M0, W0, c
if inputed_network:
    subfolder = 'initial_network'
    initial_graph = igraph.load(os.path.join(subfolder, 'g0.' + format_graph), format=format_graph)
    tech_graph = igraph.load(os.path.join(subfolder, 'tech_graph.' + format_graph), format=format_graph)
    nb_suppliers = np.array(initial_graph.degree(list(range(nb_firms)), mode="in"))
    M0 = np.array(initial_graph.get_adjacency(attribute=None).data)
    Mbar = np.array(tech_graph.get_adjacency(attribute=None).data)
    Wbar = np.array(tech_graph.get_adjacency(attribute="weight", default=0).data)
    W0 = M0 * Wbar
    c = initial_graph.ecount() / initial_graph.vcount()
    #supplier_id_list = np.fromfile(subfolder+'/'+'supplier_id_list', sep=',')
    #alternate_supplier_id_list = np.fromfile(subfolder+'/'+'alternate_supplier_id_list', sep=',')
    supplier_id_list = np.load(os.path.join(subfolder, 'supplier_id_list.npy'), allow_pickle=True)
    alternate_supplier_id_list = np.load(os.path.join(subfolder, 'alternate_supplier_id_list.npy'), allow_pickle=True)
    if initial_graph.vcount() != nb_firms:
        print(("Inadequate inputed network: n is", nb_firms, "while g0.vcount() is", initial_graph.vcount()))

## Option 2: Generate new graphs
else:
    initial_graph = initialize_graph(nb_firms, c, topology)
    supplier_id_list, nb_suppliers = identify_suppliers(initial_graph)
    tech_graph, alternate_supplier_id_list = create_technology_graph(initial_graph, a, b, sigma_w, nb_suppliers,
                                                                     nb_extra_suppliers, supplier_id_list)
    Wbar = get_tech_matrix(tech_graph)
    M0, W0, Wbar = get_initial_matrices(initial_graph, tech_graph)

print(("a*b: max " + str(max(a * b))))
print(("(1-a)*b*wji: max " + str(((1 - a) * b * Wbar).max())))

# Export inputed network
# if export_initial_network:
#     subfolder = 'initial_network'
#     initial_graph.save(subfolder + '/' + 'g0' + '.' + format_graph, format=format_graph)
#     tech_graph.save(subfolder+'/'+'tech_graph'+'.'+format_graph, format=format_graph)
#     np.save(subfolder+'/'+'supplier_id_list', supplier_id_list)
#     np.save(subfolder+'/'+'alternate_supplier_id_list', alternate_supplier_id_list)
#     print(("Network data exported in folder:", subfolder))

# Tier
min_tier = 0
max_tier = 100  # g0.diameter()
tier = draw_random_vector_lognormal(mean_tier, sigma_tier, nb_firms, min_tier, max_tier, integer=True)
print(max_tier, tier)

# Evaluate how far firms are to cover all networks
initial_coverage = sum([len(identify_firms_within_tier(firm_id, initial_graph, tier[firm_id]))
                        for firm_id in range(nb_firms)]) / nb_firms / nb_firms
print("Coverage", initial_coverage)

# Option to shut down one firm
shot_firm = None
if randomly_shoot_one_firm:
    #alternate_supplier_id_list
    shot_firm = random.randint(0, nb_firms - 1)
    print(("The shot firm is", shot_firm))
    Wbar[:, shot_firm] = 0
    Wbar[shot_firm, :] = 0
    #W0[shot_firm, :] = 0
    #a[shot_firm] = 1
    #b[shot_firm] = 0.00001
    z[shot_firm] = 0.00001

#================================================================================
# Create specific folder to store outputs
if export:
    output_folder = create_export_folder(exp_name)
    time_series_file = initialize_ts_file(output_folder, export_prices_productions, nb_firms)
    export_parameters(output_folder, nb_rounds, topology, nb_firms, c, cc, sigma_a, sigma_b, sigma_z, sigma_w,
                      a, b, z, nb_suppliers, nb_extra_suppliers, initial_graph, tech_graph, save_networks)

    if export_who_rewires:
        rewiring_firms_file = initialize_rewiring_file(output_folder)

# Option to count the number of unique network
if count_nb_unique_ntw:
    name_Mfiles = ['M_0.txt']
    nb_unique_ntw = 1

# Initialise observables
#rewiring_ts = np.empty(1 + Tfinal) # from 0 to Tfinal. The value at 0 does not count
#rewiring_ts[0] = 0
#utility_ts = np.empty(1 + Tfinal) # from 0 to Tfinal
#rewiring_time = []

#================================================================================
# Compute initial equilibrium
W = W0.copy()
g = initial_graph.copy()

eq = compute_equilibrium(a, b, z, W, nb_firms, shot_firm)

#================================================================================
# Initialize some variables
utility_ts = calculate_utility(eq)
utility = [calculate_utility(eq)]
rewiring_ts = 0
rewiring_ts_last_round = 0
score = None
if get_score:
    score = compute_cost_gap(a, b, z, W, nb_firms, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
    min_score = score
    scores = [score]
    av_score = score

if export:
    append_ts_file(time_series_file, 0, rewiring_ts, utility_ts, eq, export_prices_productions, score)

#================================================================================
# Start time loop
t = 0
eq_reached = 0
W_last_round = W.copy()
W_last_2_round = W.copy()
W_last_3_round = W.copy()
W_last_4_round = W.copy()
W_last_5_round = W.copy()

for r in range(1, nb_rounds + 1):

    # Update the time-changing variable
    rewiring_ts_last_round = rewiring_ts
    W_last_5_round = W_last_4_round.copy()
    W_last_4_round = W_last_3_round.copy()
    W_last_3_round = W_last_2_round.copy()
    W_last_2_round = W_last_round.copy()
    W_last_round = W.copy()

    # Update the rewiring order
    rewiring_order = np.random.choice(list(range(0, nb_firms)), replace=False, size=nb_firms)

    # Loop through each firm
    print("\nRound: " + str(r))
    for i in range(nb_firms):
        t += 1
        # Select one rewiring firm
        id_rewiring_firm = rewiring_order[i]
        if id_rewiring_firm == shot_firm:
            continue

        # Update the delta W
        deltaW = W.sum(axis=0) - 1
        bModif = b * (1 + deltaW * (1 - a))

        # Compute current mean cost
        current_cost = eq['P'][id_rewiring_firm]
        potential_cost = current_cost

        # Loop over current and potential suppliers to evaluate the best switch
        do_rewiring = False  # flag that is turned to 1 if the firm rewire
        id_supplier_to_remove = None  # list that store the current supplier to be replaced, if any
        id_supplier_to_add = None  # list that store the current supplier to be added, if any

        # Save W
        W_last_ts = W.copy()

        # Visit one supplier
        for id_visited_supplier in alternate_supplier_id_list[id_rewiring_firm]:
            if id_visited_supplier == shot_firm:
                continue
            # profit_dic[id_visited_supplier] = {}
            # score_dic[id_visited_supplier] = {}
            W[id_visited_supplier, id_rewiring_firm] = Wbar[
                id_visited_supplier, id_rewiring_firm]  # put the i/o coef of the technological matrix

            # And try to remove one of its current supplier
            for id_replaced_supplier in supplier_id_list[id_rewiring_firm]:
                # print('test', id_rewiring_firm, id_replaced_supplier, id_visited_supplier)
                W[id_replaced_supplier, id_rewiring_firm] = 0  # on enleve ce lien dans le W
                # If firms are myopic, they anticipate their new profit based on the current equilibrium
                # they do not take into account the impact of their rewiring on the system
                if myopic:
                    # need to update g igraph object so that we can apply the neighboorhood function
                    g.delete_edges([(id_replaced_supplier, id_rewiring_firm)])
                    g.add_edge(id_visited_supplier, id_rewiring_firm)
                    firms_within_tiers = identify_firms_within_tier(id_rewiring_firm, g, tier[id_rewiring_firm])
                    g.delete_edges([(id_visited_supplier, id_rewiring_firm)])
                    g.add_edge(id_replaced_supplier, id_rewiring_firm)
                    partial_eq, estimated_new_cost = compute_partial_equilibrium_and_cost(a, b, z, W, nb_firms,
                                                                                          eq,
                                                                                          firms_within_tiers,
                                                                                          id_rewiring_firm,
                                                                                          shot_firm)
                # otherwise firms have a perfect anticipation
                else:
                    new_eq = compute_equilibrium(a, b, z, W, nb_firms, shot_firm)
                    estimated_new_cost = new_eq['P'][id_rewiring_firm]

                if estimated_new_cost < potential_cost - EPSILON:
                    potential_cost = estimated_new_cost
                    if myopic:  # if myopic, the realized full equilibrium is computed after rewiring is done
                        new_eq = compute_equilibrium(a, b, z, W, nb_firms, shot_firm)

                    eq = new_eq
                    do_rewiring = True
                    id_supplier_toremove = id_replaced_supplier
                    id_supplier_toadd = id_visited_supplier

                # Apres le test d'un supplier a remplacer, on remet le lien dans W
                W[id_replaced_supplier, id_rewiring_firm] = Wbar[id_replaced_supplier, id_rewiring_firm]

                # Apres le test du nouveau supplier, on remet le lien dans W
            W[id_visited_supplier, id_rewiring_firm] = 0  # a la fin du test, on remet W comme avant

        if do_rewiring:  # si jamais c'est bon, on remplace pour de bon
            print(
                "Firm " + str(id_rewiring_firm),
                "changed supplier " + str(id_supplier_toremove) + " to supplier " + str(id_supplier_toadd),
                "cost decrease is " + str(current_cost - potential_cost)
            )
            if id_supplier_toadd == shot_firm:
                print('ERROR: adding the deleted firm')
                exit()
            g.delete_edges([(id_supplier_toremove, id_rewiring_firm)])
            g.add_edge(id_supplier_toadd, id_rewiring_firm)
            W[id_supplier_toadd, id_rewiring_firm] = Wbar[id_supplier_toadd, id_rewiring_firm]
            W[id_supplier_toremove, id_rewiring_firm] = 0
            supplier_id_list[id_rewiring_firm].remove(id_supplier_toremove)
            supplier_id_list[id_rewiring_firm].append(id_supplier_toadd)
            alternate_supplier_id_list[id_rewiring_firm].remove(id_supplier_toadd)
            alternate_supplier_id_list[id_rewiring_firm].append(id_supplier_toremove)
            if get_score:
                score = compute_cost_gap(a, b, z, W, n, Wbar, supplier_id_list, alternate_supplier_id_list, shot_firm)
                # print(score, min_score)
                if score < min_score:
                    min_score = score
                    # print("min_score:", min_score)
            if export:
                if export_who_rewires:
                    rewiring_firms_file.write(str(r) + ' ' + str(t) + ' ' + str(id_rewiring_firm) + "\n")

        # record score
        if get_score:
            scores += [score]
            if len(scores) > n * scores_window:
                scores.pop(0)
            av_score = np.mean(scores)

        # Update observables
        rewiring_ts = rewiring_ts + do_rewiring
        utility_ts = calculate_utility(eq)

        if export & save_networks & (do_rewiring == 1):
            # rewiring_time.append(t);
            output_filepath = os.path.join(output_folder, f"M_{t}.txt")
            np.array(g.get_edgelist()).tofile(output_filepath, sep=" ")
            if count_nb_unique_ntw:
                new_ntw = True
                for Mfile in name_Mfiles[::-1]:
                    comp = subprocess.check_output('cmp --silent M_0.txt M_final.txt || echo 1', shell=True)
                    if len(comp) == 0:
                        new_ntw = False
                        break
                if new_ntw:
                    nb_unique_ntw += 1
                name_Mfiles += ['M_' + str(t)]

        # Export economic variables
        if export:
            append_ts_file(time_series_file, t, rewiring_ts, utility_ts, eq, export_prices_productions, score)

    # Stop condition
    if apply_stop_condition:
        if rewiring_ts == rewiring_ts_last_round:
            eq_reached = 1
            print(f"Network equilibrium reached after {r - 1} rounds and {rewiring_ts} rewirings.")
            break
        if np.all(W == W_last_2_round) & np.all(W_last_round == W_last_3_round) \
                & np.all(W_last_2_round == W_last_4_round) & np.all(W_last_3_round == W_last_5_round):
            eq_reached = 2
            print(f"Limit cycle of period 2 reached after {r - 1} rounds and {rewiring_ts} rewirings.")
            break
        #if np.all(W == W_last_3_round):
        #    eq_reached = 3
        #    print("Limit cycle of period 3 reached after", r-1, "turns.")
        #    break
    utility += [calculate_utility(eq)]

if t == nb_rounds * nb_firms:
    eq_reached = 0
    print("Network equilibrium not reached")

total_time = (datetime.now() - starting_time).total_seconds()
print(f"Initial utility: {utility[0]}; final: {utility[-1]}. "
      f"Relative change: {(utility[-1] - utility[0]) / abs(utility[0])}")

if export_final_network:
    subfolder = 'initial_network'
    g.save(os.path.join(subfolder, 'g0.' + format_graph), format=format_graph)
    tech_graph.save(os.path.join(subfolder, 'tech_graph.' + format_graph), format=format_graph)
    np.save(os.path.join(subfolder, 'supplier_id_list'), supplier_id_list)
    np.save(os.path.join(subfolder, 'alternate_supplier_id_list'), alternate_supplier_id_list)
    print(("Final network data exported in folder:", subfolder))

if export:
    time_series_file.close()
    export_summary_file(output_folder, eq_reached, t, r, total_time)
    export_final_matrix(output_folder, g)
    if export_who_rewires:
        rewiring_firms_file.close()
        rewiring_firms_file = pd.read_csv(os.path.join(output_folder, "who_rewire.txt"), sep=" ")
        nb_rewiring_per_firm = rewiring_firms_file['who_rewire'].value_counts()
        nb_rewiring_per_firm_all_firm = pd.DataFrame({"firm": list(range(nb_firms)), "nb_rewirings": 0})
        nb_rewiring_per_firm_all_firm['nb_rewirings'] = nb_rewiring_per_firm_all_firm['firm'].map(nb_rewiring_per_firm)
        nb_rewiring_per_firm_all_firm['nb_rewirings'] = nb_rewiring_per_firm_all_firm['nb_rewirings'].fillna(0)
    print('Files exported in ' + output_folder)

if simple_export:
    to_export = [
        nb_firms, c, cc, sigma_w, sigma_z, sigma_b, topology, g.diameter(), np.mean(tier),
        eq_reached, r, rewiring_ts, total_time
    ]
    if get_score:
        to_export += [min_score, av_score]
    to_export = [str(x) for x in to_export]
    simple_export_filename = os.path.join(general_output_folder,
                                          "simple_dyn_results" + str(simple_export_suffix) + ".txt")
    with open(simple_export_filename, "a") as myfile:
        myfile.write(" ".join(to_export) + "\n")

if export_initntw_experiment:
    simple_export_filename = os.path.join(general_output_folder, "initntw_experiment.txt")
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f")
    with open(simple_export_filename, "a") as myfile:
        myfile.write(
            str(n) + ' ' + str(c) + ' ' + str(cc) + ' '
            + str(sigma_a) + ' ' + str(sigma_b) + ' ' + str(sigma_z) + ' ' + str(sigma_w) + ' ' + topology + ' '
            + str(eq_reached) + ' ' + str(r) + ' ' + str(total_time) + ' ' + str(current_time) + "\n"
        )
    np.array(g.get_edgelist()).tofile(os.path.join(general_output_folder, 'Mf_' + current_time + '.txt'), sep=" ")
    #if inputed_network == False: #if it is a new initial network save the initial network
    np.array(initial_graph.get_edgelist()).tofile(os.path.join(general_output_folder, 'M0_' + current_time + '.txt'),
                                                  sep=" ")

if save_networks and compute_distance_matrix:
    res = compute_distance_matrix(output_folder)
    res.to_csv(os.path.join(output_folder, "distanceMatrix.csv"))
    # os.system('cd '+output_folder+'; rm M*.txt;')
