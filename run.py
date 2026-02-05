import argparse
import copy
import os
import pickle
import shutil
import subprocess

import igraph
import numpy as np
import pandas as pd
import yaml

from export import create_export_folder, initialize_ts_file, initialize_rewiring_file, export_parameters, \
    append_ts_file, export_summary_file, export_final_matrix, create_general_output_folder
from functions import draw_random_vector_normal, compute_cost_gap, identify_firms_within_tier, \
    draw_random_vector_lognormal, compute_equilibrium, compute_partial_equilibrium_and_cost, calculate_utility, \
    get_alpha, build_W_star_from_best_sets
from generate_network import initialize_graph, create_technology_graph, identify_suppliers, get_tech_matrix, \
    get_initial_matrices, get_AiSi_productivities, compute_adjusted_z, regenerate_network_from_cached_parameters
from parameters import *


def load_config(config_path='config.yml'):
    """Load configuration from YAML file if it exists"""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            print(f"Loaded configuration from {config_path}")
            return config if config else {}
    return {}


def generate_economic_parameter(param_config, n, cli_sigma=None, param_name='param'):
    """
    Generate economic parameter vector based on mode configuration

    Args:
        param_config: Dictionary with mode and parameters
        n: Number of firms
        cli_sigma: Command-line sigma (overrides config if provided)
        param_name: Parameter name for logging (b, z, etc.)

    Returns:
        numpy array of parameter values
    """
    mode = param_config.get('mode', 'normal')

    # Command-line sigma forces normal mode
    if cli_sigma is not None and cli_sigma != 0:
        mode = 'normal'
        sigma = cli_sigma
        mean = param_config.get('mean', 1.0)
        bound_min = param_config.get('bound_min', 0.0)
        bound_max = param_config.get('bound_max', 1.0)
        values = draw_random_vector_normal(mean, sigma, n, bound_min, bound_max)
        print(f"{param_name} (CLI normal): min {min(values):.4f}, max {max(values):.4f}, mean {np.mean(values):.4f}")
        return values

    if mode == 'homogeneous':
        value = param_config.get('value', 1.0)
        values = np.full(n, value)
        print(f"{param_name} (homogeneous): value {value}")
        return values

    elif mode == 'uniform':
        min_val = param_config.get('min', 0.0)
        max_val = param_config.get('max', 1.0)
        values = np.random.uniform(min_val, max_val, n)
        print(f"{param_name} (uniform): min {min(values):.4f}, max {max(values):.4f}, mean {np.mean(values):.4f}")
        return values

    elif mode == 'normal':
        mean = param_config.get('mean', 1.0)
        sigma = param_config.get('sigma', 0.0)
        bound_min = param_config.get('bound_min', 0.0)
        bound_max = param_config.get('bound_max', 1.0)
        values = draw_random_vector_normal(mean, sigma, n, bound_min, bound_max)
        print(f"{param_name} (normal): min {min(values):.4f}, max {max(values):.4f}, mean {np.mean(values):.4f}")
        return values

    else:
        raise ValueError(f"Unknown mode '{mode}' for parameter {param_name}")


def parse_arguments():
    """Parse command line arguments using argparse and return args with economic config"""
    # Load config file first to get defaults
    config = load_config()

    parser = argparse.ArgumentParser(description='Network rewiring simulation')

    # Required arguments
    parser.add_argument('--exp-type', type=str, required=False,
                        default=config.get('exp_type', 'ts'),
                        help='Experiment type (e.g., ts, initntw, hetero)')
    parser.add_argument('--nb-rounds', type=int, required=False,
                        default=config.get('nb_rounds', 10),
                        help='Number of simulation rounds')
    parser.add_argument('--nb-firms', type=int, required=False,
                        default=config.get('nb_firms', 20),
                        help='Number of firms in the network')
    parser.add_argument('--cc', type=int, required=False,
                        default=config.get('cc', 4),
                        help='Number of connections per firm')

    # Sigma parameters
    parser.add_argument('--sigma-w', type=float, required=False,
                        default=config.get('sigma_w', 0),
                        help='Sigma for weights')
    parser.add_argument('--sigma-z', type=float, required=False,
                        default=config.get('sigma_z', 0),
                        help='Sigma for productivity')
    parser.add_argument('--sigma-b', type=float, required=False,
                        default=config.get('sigma_b', 0),
                        help='Sigma for returns to scale')
    parser.add_argument('--sigma-a', type=float, required=False,
                        default=config.get('sigma_a', 0),
                        help='Sigma for labor share')

    # Other parameters
    parser.add_argument('--aisi-spread', type=float, required=False,
                        default=config.get('aisi_spread', 0),
                        help='AiSi productivity spread parameter')
    parser.add_argument('--network-type', type=str, required=False,
                        default=config.get('network_type', 'new_tech'),
                        choices=['new_tech', 'same_all', 'same_tech_new_init'],
                        help='Network generation mode: new_tech (generate new), same_all (load cached), same_tech_new_init (cached params, new topology)')
    parser.add_argument('--exp-name', type=str, required=False,
                        default=config.get('exp_name', 'default_exp'),
                        help='Experiment name for output files')
    parser.add_argument('--anticipation-mode', type=str, required=False,
                        default=config.get('anticipation_mode', 'full'),
                        choices=['full', 'partial', 'no_anticipation', "aa"],
                        help='Anticipation mode: full (compute full equilibrium), partial (compute partial equilibrium within tier distance), or no_anticipation (use current prices without recomputing equilibrium)')
    parser.add_argument('--tier', type=int, required=False,
                        default=config.get('tier', 0),
                        help='Tier parameter: neighborhood distance for partial anticipation mode (only used when anticipation-mode is partial)')
    parser.add_argument('--export-initntw', action='store_true',
                        help='Export init_ntw experiment data (sets export_initntw_experiment=True)')

    args = parser.parse_args()

    # Attach economic_params config to args for later use
    args.economic_params = config.get('economic_params', {})

    return args


def get_job_specific_tmp_dir(exp_name, nb_firms, cc, AiSi_spread):
    """Generate unique tmp directory name for this job to avoid conflicts in parallel execution"""
    # Try SLURM_JOB_ID first, fallback to timestamp if not available
    job_id = os.environ.get('SLURM_JOB_ID', datetime.now().strftime("%Y%m%d_%H%M"))

    # For init_ntw_launcher pattern, use experiment name to share cache across runs
    # Check for common init_ntw experiment names (exact match or with suffix)
    init_ntw_patterns = ['testee', 'no_anticipation', 'partial', 'full', "aa"]
    if exp_name in init_ntw_patterns or any(exp_name.startswith(pattern + '_') for pattern in init_ntw_patterns):
        # Use exp_name directly to allow different iterations to have different caches
        job_id = f"{exp_name}_{nb_firms}_{cc}_{AiSi_spread}"

    tmp_dir = f'tmp_{job_id}'

    # Create directory if it doesn't exist
    os.makedirs(tmp_dir, exist_ok=True)

    return tmp_dir


# Parse command line arguments
args = parse_arguments()

# Get job-specific tmp directory (used for both network files and parameters)
TMP_DIR = get_job_specific_tmp_dir(args.exp_name, args.nb_firms, args.cc, args.aisi_spread)

# Model parameters from command line
sigma_w = args.sigma_w
sigma_z = args.sigma_z
sigma_b = args.sigma_b
sigma_a = args.sigma_a
AiSi_spread = args.aisi_spread
# Override export_initntw_experiment if command-line flag is set
if args.export_initntw:
    export_initntw_experiment = True

# Set anticipation mode
anticipation_mode = args.anticipation_mode
aa_mode = (anticipation_mode == 'aa')

if anticipation_mode == 'partial':
    partial_anticipation = True
    no_anticipation = False
    mean_tier = args.tier
elif anticipation_mode == 'full':
    partial_anticipation = False
    no_anticipation = False
    mean_tier = args.tier  # unused
elif anticipation_mode == 'no_anticipation':
    partial_anticipation = False
    no_anticipation = True
    mean_tier = args.tier  # unused
elif anticipation_mode == 'aa':
    partial_anticipation = False
    no_anticipation = False
    mean_tier = args.tier  # unused
else:
    raise ValueError(f"Unknown anticipation mode: {anticipation_mode}. Use 'full', 'partial', or 'no_anticipation'.")

update_eq_mode = 'after_each_rewiring'  # after_each_rewiring or after_each_round or at_the_end (unused with aa)

rewiring_test = 'try_all'

# Network generation mode handling
if args.network_type == "same_all":
    inputed_network = True
    regenerate_from_cache = False
    novelty_suffix = ""
elif args.network_type == "same_tech_new_init":
    inputed_network = True  # Load cached parameters
    regenerate_from_cache = True  # But regenerate network topology
    novelty_suffix = '_REGEN'
else:  # new_tech
    inputed_network = False
    regenerate_from_cache = False
    novelty_suffix = '_NEWNET'

exp_name = args.exp_name

# Do runs
nb_rounds = args.nb_rounds
nb_firms = args.nb_firms
c = 4
cc = args.cc
print('cc', cc)

# Override exp_type from parameters.py with command line argument
exp_type = args.exp_type

#================================================================================
# Retrieve parameter name
general_output_folder = create_general_output_folder(exp_type, exp_name)
# Start counting time
starting_time = datetime.now()

#================================================================================
# Parameters that need n
wealth = nb_firms
nb_extra_suppliers = np.full(nb_firms, cc)

if inputed_network:
    cache_initial_network = False
    cache_firm_parameters = False
    a = pickle.load(open(os.path.join(TMP_DIR, 'a'), 'rb'))
    b = pickle.load(open(os.path.join(TMP_DIR, 'b'), 'rb'))
    z = pickle.load(open(os.path.join(TMP_DIR, 'z'), 'rb'))
else:  # new_tech mode
    cache_initial_network = True
    cache_firm_parameters = True

    # Get economic parameters configuration
    econ_params = args.economic_params

    # Economic parameters: returns to scale b
    b_config = econ_params.get('b', {})
    # Command-line sigma_b overrides config
    cli_sigma_b = sigma_b if sigma_b != 0 else None
    b = generate_economic_parameter(b_config, nb_firms, cli_sigma_b, param_name='b')

    # Economic parameters: labor share a
    a_config = econ_params.get('a', {})
    mean_a = a_config.get('mean', 0.5)
    eps_a = a_config.get('eps', 0.05)
    min_a = eps_a
    # Use command-line sigma_a if provided, otherwise use config sigma_a (default 0)
    effective_sigma_a = sigma_a if sigma_a != 0 else a_config.get('sigma', 0)
    # Generate a with per-firm max based on b values
    a = np.array(
        [draw_random_vector_normal(mean_a, effective_sigma_a, nb_firms, min_a, min((1 - eps_a) / item, 1 - eps_a))[0]
         for item in list(b)])
    print(f"a: min {min(a):.4f}, max {max(a):.4f}, mean {np.mean(a):.4f}")

    # Economic parameters: Productivity z
    z_config = econ_params.get('z', {})
    # Command-line sigma_z overrides config
    cli_sigma_z = sigma_z if sigma_z != 0 else None
    z = generate_economic_parameter(z_config, nb_firms, cli_sigma_z, param_name='z')

if cache_firm_parameters:
    pickle.dump(a, open(os.path.join(TMP_DIR, 'a'), 'wb'))
    pickle.dump(b, open(os.path.join(TMP_DIR, 'b'), 'wb'))
    pickle.dump(z, open(os.path.join(TMP_DIR, 'z'), 'wb'))

#================================================================================
# Create tech network Wbar and initial input-output network W0
## Option 1: Load existing network. needs g0, techgraph, M0, W0, c
if inputed_network:
    subfolder = TMP_DIR
    initial_graph = igraph.load(os.path.join(subfolder, 'g0.' + format_graph), format=format_graph)
    if initial_graph.vcount() != nb_firms:
        print(("Inadequate inputed network: n is", nb_firms, "while g0.vcount() is", initial_graph.vcount()))
        print(list(range(nb_firms)))
        print([v.index for v in initial_graph.vs])
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
    AiSi = np.load(os.path.join(subfolder, 'AiSi.npy'), allow_pickle=True)
    
    # If regenerate_from_cache, create new network topology from cached parameters
    if regenerate_from_cache:
        print("Regenerating network topology from cached parameters...")
        regen_result = regenerate_network_from_cached_parameters(a, b, z, Wbar, AiSi)
        supplier_id_list = regen_result['supplier_id_list']
        alternate_supplier_id_list = regen_result['alternate_supplier_id_list'] 
        initial_graph = regen_result['initial_graph']
        M0 = regen_result['M0']
        W0 = regen_result['W0']

## Option 2: Generate new graphs
else:
    initial_graph = initialize_graph(nb_firms, c, topology)
    supplier_id_list, nb_suppliers = identify_suppliers(initial_graph)
    tech_graph, alternate_supplier_id_list = create_technology_graph(initial_graph, a, b, sigma_w, nb_suppliers,
                                                                     nb_extra_suppliers, supplier_id_list)
    Wbar = get_tech_matrix(tech_graph)
    M0, W0, Wbar = get_initial_matrices(initial_graph, tech_graph)
    AiSi = get_AiSi_productivities(supplier_id_list, alternate_supplier_id_list, spread=AiSi_spread)

print(("a*b: max " + str(max(a * b))))
print(("(1-a)*b*wji: max " + str(((1 - a) * b * Wbar).max())))

# Export inputed network
if export_initial_network:
    if not inputed_network:
        subfolder = TMP_DIR
        initial_graph.save(subfolder + '/' + 'g0' + '.' + format_graph, format=format_graph)
        tech_graph.save(subfolder + '/' + 'tech_graph' + '.' + format_graph, format=format_graph)
        np.save(subfolder + '/' + 'supplier_id_list', supplier_id_list)
        np.save(subfolder + '/' + 'alternate_supplier_id_list', alternate_supplier_id_list)
        np.save(subfolder + '/' + 'AiSi', AiSi)
        print(("Network data exported in folder:", subfolder))

# Tier: defines neighborhood distance for partial anticipation mode
# In full anticipation mode, tier is computed but not used
min_tier = 0
max_tier = 100  # g0.diameter()
tier = draw_random_vector_lognormal(mean_tier, sigma_tier, nb_firms, min_tier, max_tier, integer=True)
print(f"Anticipation mode: {anticipation_mode}, Mean tier: {mean_tier}, Tier values: {tier}")

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

adjusted_z = compute_adjusted_z(z, AiSi, supplier_id_list)
eq = compute_equilibrium(a, b, adjusted_z, W, nb_firms, shot_firm)

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

if aa_mode:
    P_ref = eq['P'].copy()  # all firms evaluate using the same prices
    aa_best_supplier_set = supplier_id_list.copy()
    aa_best_cost = P_ref.copy()

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
    # rewiring_order = [7, 9, 12, 5, 8, 0, 4, 19, 15, 6, 13, 16, 1, 18, 17, 11, 2, 10, 3, 14]
    # Loop through each firm
    print("\nRound: " + str(r))

    # --- AA ratchet: freeze prices within the round ---
    if aa_mode:
        # store best action per firm (synchronous update)
        aa_do = np.zeros(nb_firms, dtype=bool)
        aa_remove = np.full(nb_firms, -1, dtype=int)
        aa_add = np.full(nb_firms, -1, dtype=int)
        aa_adjusted_z = compute_adjusted_z(z, AiSi, aa_best_supplier_set)

    for i in range(nb_firms):
        t += 1
        # Select one rewiring firm
        id_rewiring_firm = rewiring_order[i]
        if id_rewiring_firm == shot_firm:
            continue

        # Update the delta W
        # deltaW = W.sum(axis=0) - 1
        # bModif = b * (1 + deltaW * (1 - a))
        current_adjusted_z_i = compute_adjusted_z(z, AiSi, supplier_id_list)[id_rewiring_firm]

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
            alpha = get_alpha(a, W)
            # profit_dic[id_visited_supplier] = {}
            # score_dic[id_visited_supplier] = {}
            W[id_visited_supplier, id_rewiring_firm] = Wbar[
                id_visited_supplier, id_rewiring_firm]  # put the i/o coef of the technological matrix

            # And try to remove one of its current supplier
            for id_replaced_supplier in supplier_id_list[id_rewiring_firm]:
                # print('test', id_rewiring_firm, id_replaced_supplier, id_visited_supplier)
                W[id_replaced_supplier, id_rewiring_firm] = 0  # on enleve ce lien dans le W
                # Compute adjusted productivity based on supplier changes
                tmp_supplier_id_list = copy.deepcopy(supplier_id_list)
                tmp_supplier_id_list[id_rewiring_firm].remove(id_replaced_supplier)
                tmp_supplier_id_list[id_rewiring_firm].append(id_visited_supplier)
                adjusted_z = compute_adjusted_z(z, AiSi, tmp_supplier_id_list)

                if no_anticipation:
                    # No anticipation mode: use current equilibrium prices to naively estimate cost
                    # The firm does NOT recompute equilibrium, just uses current prices with new supplier structure
                    # share_inputs_from_old_supplier = (Wbar[id_replaced_supplier, id_rewiring_firm]
                    #                                   * (1 - a[id_rewiring_firm])
                    #                                   * eq['P'][id_rewiring_firm]) \
                    #                                  / (eq['P'][id_replaced_supplier] * alpha[id_rewiring_firm])
                    # current_cost = eq['P'][id_rewiring_firm]
                    # delta_price = eq['P'][id_visited_supplier] - eq['P'][id_replaced_supplier]
                    # estimated_new_cost = current_cost + share_inputs_from_old_supplier * delta_price

                    #other method
                    estimated_new_cost = (np.prod(np.power(eq['P'],
                                                           (1-a[id_rewiring_firm]) * W[:, id_rewiring_firm]))
                                          / current_adjusted_z_i)

                elif partial_anticipation:
                    # Partial anticipation mode: firms only see within tier[i] distance
                    # Update graph to identify neighborhood, compute partial equilibrium
                    g.delete_edges([(id_replaced_supplier, id_rewiring_firm)])
                    g.add_edge(id_visited_supplier, id_rewiring_firm)
                    firms_within_tiers = identify_firms_within_tier(id_rewiring_firm, g, tier[id_rewiring_firm])
                    g.delete_edges([(id_visited_supplier, id_rewiring_firm)])
                    g.add_edge(id_replaced_supplier, id_rewiring_firm)
                    partial_eq, estimated_new_cost = compute_partial_equilibrium_and_cost(a, b, adjusted_z, W,
                                                                                          firms_within_tiers,
                                                                                          id_rewiring_firm,
                                                                                          shot_firm)
                elif aa_mode:
                    # AA: evaluate best response at frozen prices P_ref (no GE recomputation here)
                    potential_cost = aa_best_cost[id_rewiring_firm]
                    estimated_new_cost = (np.prod(np.power(P_ref, (1 - a[id_rewiring_firm]) * W[:, id_rewiring_firm]))
                                          / aa_adjusted_z[id_rewiring_firm])

                else:  # full anticipation
                    # Full anticipation mode: firms compute full equilibrium
                    adjusted_z = compute_adjusted_z(z, AiSi, tmp_supplier_id_list)
                    new_eq = compute_equilibrium(a, b, adjusted_z, W, nb_firms, shot_firm)
                    estimated_new_cost = new_eq['P'][id_rewiring_firm]
                    # print(f"Firm {id_rewiring_firm} replacing {id_replaced_supplier} by {id_visited_supplier} "
                    #       f"estimated_new_cost: {estimated_new_cost:.3f}; "
                    #       f"current_cost: {current_cost:.3f}; "
                    #       f"potential_cost: {potential_cost:.3f}")

                # print(f'Firm {id_rewiring_firm} '
                #       f'replacing {id_replaced_supplier} (AiSi={current_adjusted_z_i:.03f}, p={eq['P'][id_replaced_supplier]:.03f}) '
                #       f'by {id_visited_supplier} (AiSi={adjusted_z[id_rewiring_firm]:.03f}, p={eq['P'][id_visited_supplier]:.03f}): '
                #       f'current cost {current_cost:.03f}, new estimated cost {estimated_new_cost:.03f}')


                if estimated_new_cost < potential_cost - EPSILON:
                    # print("    DO REWIRING")
                    do_rewiring = True
                    potential_cost = estimated_new_cost
                    id_supplier_to_remove = id_replaced_supplier
                    id_supplier_to_add = id_visited_supplier

                if aa_mode and do_rewiring:
                    aa_do[id_rewiring_firm] = True
                    aa_remove[id_rewiring_firm] = id_supplier_to_remove
                    aa_add[id_rewiring_firm] = id_supplier_to_add
                    aa_best_cost[id_rewiring_firm] = estimated_new_cost
                    aa_best_supplier_set[id_rewiring_firm] = sorted(list((set(supplier_id_list[id_rewiring_firm])
                                                                  | {id_supplier_to_add}) - {id_supplier_to_remove}))
                    # P_ref[id_rewiring_firm] = potential_cost

                    # IMPORTANT: do not apply rewiring now in AA mode
                    do_rewiring = False

                # Apres le test d'un supplier à remplacer, on remet le lien dans W
                W[id_replaced_supplier, id_rewiring_firm] = Wbar[id_replaced_supplier, id_rewiring_firm]

                # Apres le test du nouveau supplier, on remet le lien dans W
            W[id_visited_supplier, id_rewiring_firm] = 0  # a la fin du test, on remet W comme avant

        if do_rewiring:  # si jamais c'est bon, on remplace pour de bon
            print(f"    Firm {id_rewiring_firm} replaces supplier {id_supplier_to_remove} "
                  f"by {id_supplier_to_add}, cost decrease is {current_cost - potential_cost}")
            if id_supplier_to_add == shot_firm:
                print('    ERROR: adding the deleted firm')
                exit()
            g.delete_edges([(id_supplier_to_remove, id_rewiring_firm)])
            g.add_edge(id_supplier_to_add, id_rewiring_firm)
            W[id_supplier_to_add, id_rewiring_firm] = Wbar[id_supplier_to_add, id_rewiring_firm]
            W[id_supplier_to_remove, id_rewiring_firm] = 0
            supplier_id_list[id_rewiring_firm].remove(id_supplier_to_remove)
            supplier_id_list[id_rewiring_firm].append(id_supplier_to_add)
            alternate_supplier_id_list[id_rewiring_firm].remove(id_supplier_to_add)
            alternate_supplier_id_list[id_rewiring_firm].append(id_supplier_to_remove)

            if not aa_mode:
                # For partial and no anticipation: compute realized full equilibrium after rewiring is done
                adjusted_z = compute_adjusted_z(z, AiSi, supplier_id_list)
                new_eq = compute_equilibrium(a, b, adjusted_z, W, nb_firms, shot_firm)
            if update_eq_mode == "after_each_rewiring":
                print(f"    Last price: {eq['P'][id_rewiring_firm]:.3f}; "
                      f"new price: {new_eq['P'][id_rewiring_firm]:.3f}")
                eq = new_eq

            if get_score:
                score = compute_cost_gap(a, b, adjusted_z, W, n, Wbar, supplier_id_list, alternate_supplier_id_list,
                                         shot_firm)
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

    if update_eq_mode == "after_each_round":
        eq = new_eq

    # --- AA ratchet: apply all rewires simultaneously, then recompute GE once ---
    if aa_mode:
        #     any_rewire = aa_do.any()
        #     if any_rewire:
        #         for firm_id in np.where(aa_do)[0]:
        # if firm_id == shot_firm:
        #     continue
        # rem = aa_remove[firm_id]
        # add = aa_add[firm_id]
        # if rem < 0 or add < 0:
        #     continue
        #
        # print(
        #     f"    [AA] Firm {firm_id} replaces supplier {rem} by {add}, "
        #     f"estimated cost decrease is {eq['P'][firm_id] - aa_best_cost[firm_id]}"
        # )
        #
        # # apply to graph + matrices + lists
        # g.delete_edges([(rem, firm_id)])
        # g.add_edge(add, firm_id)
        #
        # W[add, firm_id] = Wbar[add, firm_id]
        # W[rem, firm_id] = 0
        #
        # supplier_id_list[firm_id].remove(rem)
        # supplier_id_list[firm_id].append(add)
        # alternate_supplier_id_list[firm_id].remove(add)
        # alternate_supplier_id_list[firm_id].append(rem)

        # rewiring_ts += 1  # count rewires (synchronous)
        aa_adjusted_z = compute_adjusted_z(z, AiSi, aa_best_supplier_set)
        P_next = aa_best_cost.copy()#np.minimum(P_ref, aa_best_cost)

        # stopping criterion on prices (not on rewiring_ts)
        max_rel = np.max(np.abs(P_next - P_ref) / np.maximum(P_ref, 1e-12))
        print(f"[AA] nb of swaps: {np.sum(aa_do)}")
        print(f"[AA] max rel price change = {max_rel:.3e}")
        print(P_next)
        if max_rel < 1e-10:
            print(f"[AA] Price fixed point reached after {r} rounds.")
            print(aa_do)
            print(aa_best_supplier_set)
            Wstar = build_W_star_from_best_sets(aa_best_supplier_set, Wbar)
            aa_adjusted_z = compute_adjusted_z(z, AiSi, aa_best_supplier_set)
            final_eq = compute_equilibrium(a, b, aa_adjusted_z, Wstar, nb_firms, shot_firm)
            print(P_next)
            print(final_eq['P'])
            break

        current_Wbest = build_W_star_from_best_sets(aa_best_supplier_set, Wbar)
        current_eq = compute_equilibrium(a, b, aa_adjusted_z, current_Wbest, nb_firms, shot_firm)
        print(aa_adjusted_z)
        # print(W)
        P_ge = current_eq['P']
        P_ratchet = P_next  # or P_next if you already updated
        rel_err = (P_ge - P_ratchet) / P_ratchet
        print(f"[AA diag] max rel diff |P_ge - P_ratchet| / P_ratchet = "
              f"{np.max(np.abs(rel_err)):.3e}")
        P_ref = P_next


    # Stop condition
    if apply_stop_condition and not aa_mode:
        if rewiring_ts == rewiring_ts_last_round:
            eq_reached = 1
            print(f"Network equilibrium reached after {r - 1} rounds and {rewiring_ts} rewirings.")
            print([sorted(l) for l in supplier_id_list])
            adjusted_z = compute_adjusted_z(z, AiSi, supplier_id_list)
            final_eq = compute_equilibrium(a, b, adjusted_z, W, nb_firms, shot_firm)
            print(final_eq)
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

if get_last_score:
    adjusted_z = compute_adjusted_z(z, AiSi, supplier_id_list)
    av_score = compute_cost_gap(a, b, adjusted_z, W, nb_firms, Wbar, supplier_id_list, alternate_supplier_id_list,
                                shot_firm)
    min_score = av_score

total_time = (datetime.now() - starting_time).total_seconds()
print(f"Initial utility: {utility[0]}; final: {utility[-1]}. "
      f"Relative change: {(utility[-1] - utility[0]) / abs(utility[0])}")

if export_final_network:
    subfolder = TMP_DIR
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
    if get_score or get_last_score:
        to_export += [min_score, av_score]
    to_export = [str(x) for x in to_export]
    simple_export_filename = os.path.join(general_output_folder,
                                          "simple_dyn_results" + str(simple_export_suffix) + ".txt")
    with open(simple_export_filename, "a") as myfile:
        myfile.write(" ".join(to_export) + "\n")

if export_initntw_experiment:
    simple_export_filename = os.path.join(general_output_folder, "initntw_experiment.txt")
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S%f")
    with open(simple_export_filename, "a") as myfile:
        myfile.write(
            str(nb_firms) + ' ' + str(c) + ' ' + str(cc) + ' '
            + str(sigma_a) + ' ' + str(sigma_b) + ' ' + str(sigma_z) + ' ' + str(sigma_w) + ' ' + str(AiSi_spread) + " "
            + topology + ' '
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

# Cleanup job-specific tmp directory unless explicitly kept
if not keep_tmp_dir and os.path.exists(TMP_DIR):
    try:
        shutil.rmtree(TMP_DIR)
        print(f"Cleaned up temporary directory: {TMP_DIR}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory {TMP_DIR}: {e}")
