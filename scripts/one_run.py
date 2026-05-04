import random
import numpy as np

from rewiring.networks import generate_base_network
from rewiring.parameters import generate_a_parameter, generate_parameter
from rewiring.simulation import run_unified_simulation
from scripts.visibility_study import build_tier_array

# Reproducibility
SEED = 42

# Economic parameters (the n=100 main-sweep operating point)
N, C, CC = 50, 4, 4
tier_mean, tier_std = 1, 1
random.seed(SEED); np.random.seed(SEED)
tier_rng = np.random.default_rng(100)
# b = generate_parameter({"mode": "uniform", "min": 0.9, "max": 1.1}, N, "b", verbose=False)
b = generate_parameter({"mode": "homogeneous", "value": 0.9}, N, "b", verbose=False)
# a = generate_a_parameter({"mode": "uniform", "min": 0.4, "max": 0.6}, b, N, verbose=False)
a = generate_a_parameter({"mode": "homogeneous", "value": 0.5}, b, N, verbose=False)
# z = generate_parameter({"mode": "uniform", "min": 0.9, "max": 1.1}, N, "z", verbose=False)
z = generate_parameter({"mode": "homogeneous", "value": 1.0}, N, "z", verbose=False)
tier = build_tier_array(N, tier_mean, tier_std, tier_rng)

# Network state (Wbar, AiSi, supplier sets, ...)
state = generate_base_network(N, C, CC, aisi_spread=0, seed=SEED, a=a, b=b, sigma_w=0)

# Run
result = run_unified_simulation(
    state, a, b, z,
    mode="limited",
    seed=SEED,
    max_swaps=1,
    nb_rounds=50,
    console_print=True,
    tier=tier
)

print({k: result[k] for k in
       ["mode", "converged", "cycle_period", "rounds",
        "total_rewirings", "initial_utility", "final_utility"]})