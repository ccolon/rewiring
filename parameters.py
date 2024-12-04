# Script parameter
from datetime import datetime

exp_name = "test"

# Model parameter
topology = "SF-FA"
sigma_w = 0.0
sigma_z = 0.0
sigma_b = 0.0
sigma_a = 0.0
rewiring_test = 'try_all'
inputed_network = False
novelty_suffix = '_NEWNET'

# Do runs
NbRound = 10
n = 20
c = 4
cc = 4

# Model option - set default parameters
myopic = False
epsilon = 0
randomly_shoot_onefirm = False
save_network_on_off = False
simple_export = False
firm_level_export = False
export = False
export_initntw_experiment = False
save_network_on_off = False
show_time = False
export_initial_network = False
export_final_network = False
export_firm_profits = False
export_who_rewires = False
compute_distance_matrix = False
simple_export_with_gini = False
simple_export_with_giniNbNet = False
apply_stop_condition = True
count_nb_unique_ntw = False
hamiltonian = False
print_score = True
tier = 0
# up to which tier should a firm have full knowledge. Valid if myopic. If tier high enough, should be the same as non-myopic

exp_type = "normal"

if exp_type == "shootInit":
    randomly_shoot_onefirm = False
    save_network_on_off = True
    export_final_network = True
    export = True

if exp_type == "shootOneSaveFinal":
    randomly_shoot_onefirm = True
    save_network_on_off = True
    export_final_network = True
    export = True

if exp_type == "recover":
    randomly_shoot_onefirm = False
    save_network_on_off = True
    export_final_network = True
    export = True

if exp_type == "shootOneRestart":
    randomly_shoot_onefirm = True
    save_network_on_off = True
    export_initial_network = True
    export_final_network = False
    export = True

if exp_type == 'initntw':
    export_initntw_experiment = True
    export_initial_network = True
    save_network_on_off = True

if exp_type == 'initntwMyopic':
    myopic = True
    export_initntw_experiment = True
    export_initial_network = True
    save_network_on_off = True

if exp_type == 'slowing':
    simple_export = True

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

if exp_type == 'heteroGini':
    simple_export_with_gini = True
    export = True
    save_network_on_off = False
    export_who_rewires = True
    count_nb_unique_ntw = False

if exp_type == 'heteroGiniEspilon':
    epsilon = 1e-6
    simple_export_with_gini = True
    export = True
    save_network_on_off = False
    export_who_rewires = True
    count_nb_unique_ntw = False

if exp_type == 'heteroGiniMyopic':
    myopic = True
    simple_export_with_gini = True
    export = True
    save_network_on_off = False
    export_who_rewires = True
    count_nb_unique_ntw = False

if exp_type == 'heteroGiniNbunique':
    simple_export_with_giniNbNet = True
    export = True
    save_network_on_off = True
    export_who_rewires = True
    count_nb_unique_ntw = True

if exp_type == "hamiltonian":
    hamiltonian = True
    apply_stop_condition = False

if exp_type == "2021_myopic":
    myopic = True
    simple_export = True

if exp_type == "tier":
    myopic = True
    simple_export = True
    tier = 1

if exp_type == 'normal':
    simple_export = True

# Saving graphs option for reuse
format_graph = 'picklez'
simple_export_suffix = '_' + datetime.now().strftime("%Y%m%d")
