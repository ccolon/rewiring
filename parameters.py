# Script parameter
from datetime import datetime

exp_name = "slowing"
exp_type = "slowing"

# Model parameter
nb_rounds = 50
nb_firms = 50
c = 4
cc = 4
myopic = False
mean_tier = 1
sigma_tier = 0
sigma_z = 0
sigma_a = 0
sigma_b = 0
sigma_w = 0
inputed_network = False
topology = "SF-FA"
randomly_shoot_one_firm = False




# Saving graphs option for reuse
format_graph = 'picklez'
simple_export_suffix = '_' + datetime.now().strftime("%Y%m%d")

# Model option - set default parameters
EPSILON = 1e-10
novelty_suffix = '_NEWNET'
save_networks = True
simple_export = False
firm_level_export = False
export = False
export_prices_productions = True
export_initntw_experiment = False
export_initial_network = False
export_final_network = True
export_who_rewires = True
compute_distance_matrix = False
apply_stop_condition = True
count_nb_unique_ntw = False
hamiltonian = False
print_score = True
get_score = False
tier = 0  # up to which tier should a firm have full knowledge. Valid if myopic. If tier high enough, should be the same as non-myopic
scores_window = 5

# up to which tier should a firm have full knowledge. Valid if myopic. If tier high enough, should be the same as non-myopic


if exp_type == "shootInit":
    randomly_shoot_one_firm = False
    save_network_on_off = True
    export_final_network = True
    export = True

if exp_type == "shootOneSaveFinal":
    randomly_shoot_one_firm = True
    save_network_on_off = True
    export_final_network = True
    export = True

if exp_type == "recover":
    randomly_shoot_one_firm = False
    save_network_on_off = True
    export_final_network = True
    export = True

if exp_type == "shootOneRestart":
    randomly_shoot_one_firm = True
    save_network_on_off = True
    export_initial_network = True
    export_final_network = False
    export = True

if exp_type == 'initntw':
    export_initntw_experiment = True
    cache_initial_network = True
    cache_firm_parameters = True
    save_network_on_off = False
    #export_final_network = True
    mean_tier = mean_tier
    sigma_tier = 0
    #export = True
    myopic = False

if exp_type == 'initntwMyopic':
    myopic = True
    export_initntw_experiment = True
    export_initial_network = True
    save_network_on_off = True

if exp_type == 'slowing':
    simple_export = True
    myopic = False

if exp_type == 'heteroSaveTs':
    simple_export = False
    export = True
    save_network_on_off = True
    export_who_rewires = True
    export_final_network = True
    compute_distance_matrix = True
    apply_stop_condition = True
    export_firm_profits = True

if exp_type == 'hetero':
    simple_export = True
    export = False
    save_network_on_off = False
    count_nb_unique_ntw = False
    apply_stop_condition = True

if exp_type == "hamiltonian":
    hamiltonian = True
    apply_stop_condition = False

if exp_type == "tier_hetero":
    myopic = True
    simple_export = True
    mean_tier = mean_tier
    sigma_tier = mean_tier

if exp_type == "tier_homo":
    myopic = True
    simple_export = True
    mean_tier = mean_tier
    sigma_tier = 0

if exp_type == 'normal':
    simple_export = True

if exp_type == "tier_ts":
    export = True
    myopic = True
    simple_export = False
    mean_tier = mean_tier
    save_network_on_off = True
    export_who_rewires = True
    export_final_network = True
    compute_distance_matrix = True
    export_firm_profits = True
    apply_stop_condition = False

