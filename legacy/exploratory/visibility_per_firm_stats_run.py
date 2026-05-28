"""Local-only per-firm rewiring stats experiment, paired with
visibility_per_firm_stats_analysis.py.

Runs 9 cells x 10 trials = 90 simulations and saves all the data needed for
the analysis (rewire events, tier_arr, cycle period, etc.) to a single
pickle file.

Cells:
    R1 (Full heterogeneity):       a U[0.4,0.6], b U[0.9,1.1], z U[0.9,1.1],
                                   aisi=0.05, sigma_w=0.05
    R2 (Firm-level het. only):     same a/b/z, aisi=0, sigma_w=0
    R3 (Homogeneous DRS):          a=hom 0.5, b=hom 0.9, z=hom 1.0,
                                   aisi=0, sigma_w=0
    R1_hetero:                     R1 economy with lognormal hetero tau

    homo tau (3 ops x 2 tau):
        R1_t0, R1_t1, R2_t0, R2_t1, R3_t0, R3_t1
    hetero tau (R1 only, 3 mean values):
        R1_h1 (mean=1, std=1), R1_h3 (mean=3, std=3), R1_h5 (mean=5, std=5)

Output: results/visibility_per_firm_stats.pkl  -- list of trial dicts.

Usage:
    python scripts/visibility_per_firm_stats_run.py
    python scripts/visibility_per_firm_stats_run.py --base_seed 5001  --output ...
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.networks import (
    generate_base_network,
    generate_random_initial_network,
)
from rewiring.simulation import run_unified_simulation


N          = 50
C          = 4
CC         = 4
MS         = 1
NB_ROUNDS  = 50
N_TRIALS   = 10

# Cell list. Each tuple: (cell_id, op_label, a_cfg, b_cfg, z_cfg, aisi, sigma_w,
#                         tier_mean, tier_std)
CELLS = [
    # 3 operating points x homo tau in {0, 1}
    ('R1_t0', 'R1: Full heterogeneity',
     'unif:0.4:0.6', 'unif:0.9:1.1', 'unif:0.9:1.1', 0.05, 0.05, 0, 0),
    ('R1_t1', 'R1: Full heterogeneity',
     'unif:0.4:0.6', 'unif:0.9:1.1', 'unif:0.9:1.1', 0.05, 0.05, 1, 0),
    ('R2_t0', 'R2: Firm-level het only',
     'unif:0.4:0.6', 'unif:0.9:1.1', 'unif:0.9:1.1', 0.0,  0.0,  0, 0),
    ('R2_t1', 'R2: Firm-level het only',
     'unif:0.4:0.6', 'unif:0.9:1.1', 'unif:0.9:1.1', 0.0,  0.0,  1, 0),
    ('R3_t0', 'R3: Homogeneous DRS',
     'hom:0.5', 'hom:0.9', 'hom:1.0',                0.0,  0.0,  0, 0),
    ('R3_t1', 'R3: Homogeneous DRS',
     'hom:0.5', 'hom:0.9', 'hom:1.0',                0.0,  0.0,  1, 0),
    # R1 hetero tau, lognormal mean = std
    ('R1_h1', 'R1: Full het, hetero tau',
     'unif:0.4:0.6', 'unif:0.9:1.1', 'unif:0.9:1.1', 0.05, 0.05, 1, 1),
    ('R1_h3', 'R1: Full het, hetero tau',
     'unif:0.4:0.6', 'unif:0.9:1.1', 'unif:0.9:1.1', 0.05, 0.05, 3, 3),
    ('R1_h5', 'R1: Full het, hetero tau',
     'unif:0.4:0.6', 'unif:0.9:1.1', 'unif:0.9:1.1', 0.05, 0.05, 5, 5),
]


def _parse_param(s, n, rng):
    parts = s.split(':')
    mode = parts[0]
    if mode == 'hom':
        return np.full(n, float(parts[1]))
    if mode == 'unif':
        return rng.uniform(float(parts[1]), float(parts[2]), n)
    raise ValueError(f"Unknown param spec: {s}")


def _build_tier_array(n, tier_mean, tier_std, rng, tier_dist='poisson'):
    """Per-firm tier array (matches visibility_study).

    Homogeneous (tier_std <= 0): constant array = round(tier_mean).
    Heterogeneous (tier_std > 0):
      tier_dist='poisson'  : tier_i ~ Poisson(lambda=tier_mean)  (default)
      tier_dist='lognormal': legacy lognormal with target mean=tier_mean,
                              std=tier_std
    """
    if tier_std <= 0:
        return np.full(n, int(round(tier_mean)), dtype=int)
    if tier_mean <= 0:
        return np.zeros(n, dtype=int)
    if tier_dist == 'poisson':
        return rng.poisson(lam=tier_mean, size=n).astype(int)
    if tier_dist == 'lognormal':
        var_n = np.log(1.0 + (tier_std / tier_mean) ** 2)
        mean_n = np.log(tier_mean) - 0.5 * var_n
        draws = rng.lognormal(mean=mean_n, sigma=np.sqrt(var_n), size=n)
        return np.clip(np.round(draws), 0, None).astype(int)
    raise ValueError(f"Unknown tier_dist={tier_dist!r}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base_seed', type=int, default=5000)
    p.add_argument('--tier_dist', type=str, default='poisson',
                   choices=['poisson', 'lognormal'],
                   help='Hetero-tau distribution (default: poisson; legacy: '
                        'lognormal).')
    p.add_argument('--output', type=str, default=None,
                   help='Output pickle path (default: results/visibility_per_firm_stats.pkl)')
    args = p.parse_args()

    out_path = (args.output or
                os.path.join(REPO_ROOT, 'results', 'visibility_per_firm_stats.pkl'))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    SEED_MOD = 2**31 - 1

    out = []
    overall_t0 = time.time()

    for cell_idx, (cell_id, op_label, a_cfg, b_cfg, z_cfg,
                   aisi, sw, tm, ts) in enumerate(CELLS):
        cell_t0 = time.time()
        print(f"\n=== Cell {cell_idx + 1}/{len(CELLS)}: {cell_id}  "
              f"({op_label};  tier mean={tm}, std={ts}) ===")

        for trial_idx in range(N_TRIALS):
            tech_seed = (args.base_seed * 1000 + cell_idx * 100 + trial_idx) % SEED_MOD
            init_seed = (tech_seed + 7919) % SEED_MOD
            tier_seed = (tech_seed + 13331) % SEED_MOD
            sim_seed  = (tech_seed + 31337) % SEED_MOD

            param_rng = np.random.default_rng(tech_seed)
            a = _parse_param(a_cfg, N, param_rng)
            b = _parse_param(b_cfg, N, param_rng)
            z = _parse_param(z_cfg, N, param_rng)

            base_state = generate_base_network(
                N, C, CC, aisi, seed=int(tech_seed), a=a, b=b, sigma_w=sw,
            )
            init_state = generate_random_initial_network(
                N, base_state['Wbar'], base_state['AiSi'], seed=int(init_seed),
            )
            tier_arr = _build_tier_array(
                N, tm, ts, np.random.default_rng(tier_seed),
                tier_dist=args.tier_dist,
            )

            t0 = time.time()
            result = run_unified_simulation(
                init_state, a, b, z, mode='limited',
                seed=int(sim_seed), max_swaps=MS, nb_rounds=NB_ROUNDS,
                tier=tier_arr, trace=True,
            )
            dt = time.time() - t0

            out.append({
                'cell_id':           cell_id,
                'op_label':          op_label,
                'tier_mean':         tm,
                'tier_std':          ts,
                'aisi':              aisi,
                'sigma_w':           sw,
                'n':                 N,
                'cc':                CC,
                'max_swaps':         MS,
                'tech_seed':         int(tech_seed),
                'init_seed':         int(init_seed),
                'sim_seed':          int(sim_seed),
                'trial_idx':         trial_idx,
                'tier_arr':          tier_arr.tolist(),
                'rounds':            int(result['rounds']),
                'cycle_period':      result.get('cycle_period'),
                'converged':         bool(result['converged']),
                'total_rewirings':   int(result['total_rewirings']),
                'per_firm_swaps':    np.asarray(result['per_firm_swaps']).tolist(),
                'rewire_events':     list(result['trace']['rewire_events']),
            })

            print(f"  trial {trial_idx + 1:>2}/{N_TRIALS}: "
                  f"rounds={result['rounds']:>3}  "
                  f"cycle_period={result.get('cycle_period')}  "
                  f"converged={result['converged']}  "
                  f"swaps={result['total_rewirings']:>5}  "
                  f"({dt:.1f}s)")

        print(f"  cell elapsed: {time.time() - cell_t0:.1f}s")

    with open(out_path, 'wb') as f:
        pickle.dump(out, f)

    print(f"\nWrote {len(out)} trials to {out_path}")
    print(f"Total elapsed: {time.time() - overall_t0:.1f}s")


if __name__ == '__main__':
    main()
