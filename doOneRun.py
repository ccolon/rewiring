from __future__ import division
# Load stuff
import sys
sys.path.insert(0,'env/site-packages')
import igraph
import pandas as pd
import numpy as np
import random
import datetime
import os
import pickle
import subprocess
import math

# Script parameter
exp_name = sys.argv[1]



# Model option - set initial parameters
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
tier = 0 # up to which tier should a firm have full knowledge. Valid if myopic. If tier high enough, should be the same as non-myopic


exp_type = sys.argv[10]

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
    tier = int(sys.argv[11])

if exp_type == 'normal':
    simple_export = True
    
    
# Saving graphs option for reuse
format_graph = 'picklez'

# Whether or not to reuse an existing graph
if sys.argv[9] == "inputed":
    inputed_network = True
    novelty_suffix = ""
else:
    inputed_network = False
    novelty_suffix = '_NEWNET'


#Load stuff
exec(open('functions.py').read())

# Model parameter
topology = "SF-FA"
sigma_w = float(sys.argv[5])
sigma_z = float(sys.argv[6])
sigma_b = float(sys.argv[7])
sigma_a = float(sys.argv[8])
rewiring_test = 'try_all'

#Do runs
NbRound = int(sys.argv[2])
n = int(sys.argv[3])
c = 4
cc = int(sys.argv[4])

simple_export_suffix = '_'+datetime.datetime.now().strftime("%Y%m%d")

# Optimization type
#if sys.argv[7] == "oneswitch":
#        exp_name = "n" + str(n) + "s" + "%04d" %int(sigma_w*1000) + '_oneswitch' + novelty_suffix
#        exec(open(general_folder + 'AllModelOneSwitch.py').read())

if hamiltonian:
    exp_name = "n" + str(n) + "s" + "%04d" %int(sigma_w*1000) + '_hamiltonian' + novelty_suffix
    exec(open('hamiltonian.py').read())

else: 
    exp_name = "n" + str(n) + "s" + "%04d" %int(sigma_w*1000) + '_fulloptim' + novelty_suffix
    exec(open('AllModelFullOptim.py').read())

    


print("________ n"+str(n)+" done")
