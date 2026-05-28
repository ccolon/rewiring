"""Per-switch cost-hurdle (chi) study (manuscript appendix: fig:switchcost).

For one parameter point in {P2, P3, P4} (labels match welfare_dispersion_study.py:
P2 = CRS+Delta_A traps, P3 = HRS structural, P4 = HRS compounded) and one
chi value, run (tech matrices) x (random initial networks) trials at n=100
and record trial-level metrics needed for the 3-panel figure:

  - configuration diversity nu_P (computed post-hoc per (point, chi, tech)
    by grouping init trials and counting unique final supplier-set
    configurations);
  - mean terminal relative cost gap theta (theta_static against the
    unconstrained best deviation -- see compute_static_gap);
  - per-firm rewiring count F = total_rewirings / n.

The seed scheme is identical to welfare_dispersion_study.py so trials with
the same (tech_seed, init_seed) share Wbar, AiSi, a, b, z, and S^(0)
across chi values -- enabling matched-pair analysis if desired.

One CSV per (point, chi, seed-batch); rows keyed by
(point, chi, tech_seed, init_seed).

Usage:
    python campaigns/switchcosts/study.py --point P2 --chi 0.005 \\
        --tech_per_job 10 --inits_per_tech 50 --base_seed 2700 \\
        --output results/switchcosts/switch_P2_chi0.005_seed2700.csv
"""
import argparse
import csv
import os
import sys

import numpy as np

# This script lives at campaigns/switchcosts/, two levels below the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.equilibrium import compute_static_gap
from rewiring.networks import (
    generate_base_network,
    generate_random_initial_network,
)
from rewiring.simulation import run_unified_simulation

# Single source of truth for parameter points: reuse the welfare-dispersion
# campaign. The switchcost figure runs only P2 / P3 / P4 -- P0 and P1 are
# skipped (P0 is the AA control, P1 is the no-friction degenerate point).
from campaigns.welfare_dispersion.study import POINTS as WELFARE_POINTS

SWITCHCOST_POINTS = {k: WELFARE_POINTS[k] for k in ('P2', 'P3', 'P4')}

N = 100
C = 4
CC = 4
A_VALUE = 0.5
Z_VALUE = 1.0
NB_ROUNDS = 200


CSV_FIELDS = [
    'point', 'chi', 'tech_seed', 'init_seed',
    'kappa',
    'converged', 'cycle_period', 'rounds', 'total_rewirings',
    'sum_p_init', 'sum_p_final_last',
    'sum_p_current', 'sum_p_static_best', 'theta_static',
    # Hash of the final supplier configuration -- enables nu_P computation
    # in the analyzer without re-storing the full edge list.
    'final_config_hash',
]


def _draw_b(point_cfg, n, rng):
    if point_cfg['b_hetero']:
        return rng.uniform(0.9, 1.1, n)
    return np.full(n, float(point_cfg['b_value']))


def _draw_z(point_cfg, n, rng):
    dz = point_cfg['delta_z']
    if dz <= 0:
        return np.full(n, Z_VALUE)
    return rng.uniform(1 - dz, 1 + dz, n)


def _config_hash(supplier_list):
    """Stable hash of the per-firm sorted supplier sets.

    Two configurations with the same (firm, supplier) edge set hash equal
    regardless of internal list order.
    """
    canonical = tuple(tuple(sorted(int(s) for s in suppliers))
                      for suppliers in supplier_list)
    return hash(canonical)


def existing_keys(path):
    if not os.path.exists(path):
        return set()
    keys = set()
    with open(path, 'r', newline='') as f:
        for row in csv.DictReader(f):
            keys.add((int(row['tech_seed']), int(row['init_seed'])))
    return keys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--point', required=True, choices=sorted(SWITCHCOST_POINTS.keys()),
                   help='Parameter point. P2 (CRS, kappa=1, Delta_A=0.05), '
                        'P3 (HRS, kappa=c\', no dispersion), or P4 '
                        '(HRS, kappa=1, compounded). Labels match '
                        'welfare_dispersion_study.py.')
    p.add_argument('--chi', type=float, required=True,
                   help='Per-switch cost hurdle (chi >= 0). 0 = baseline.')
    p.add_argument('--tech_per_job', type=int, default=10)
    p.add_argument('--inits_per_tech', type=int, default=50)
    p.add_argument('--base_seed', type=int, default=2700,
                   help='Outer seed; tech_seed = base_seed*10000 + tech_idx, '
                        'init_seed = tech_seed*100 + init_idx. Use the SAME '
                        '--base_seed across chi values to match trials.')
    p.add_argument('--nb_rounds', type=int, default=NB_ROUNDS)
    p.add_argument('--n', type=int, default=N,
                   help='Number of firms (default 100).')
    p.add_argument('--output', required=True)
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    point_cfg = SWITCHCOST_POINTS[args.point]
    SEED_MOD = 2**31 - 1
    n = args.n
    kappa = point_cfg['kappa']
    static_max_swaps = min(C, CC)  # = 4: full unconstrained C*

    print(f"Switch-cost study | point={args.point} | chi={args.chi}")
    print(f"  econ: b_hetero={point_cfg['b_hetero']} b_value={point_cfg['b_value']} "
          f"kappa={kappa} aisi={point_cfg['aisi']} sw={point_cfg['sigma_w']}")
    print(f"  budget: {args.tech_per_job} tech x {args.inits_per_tech} inits = "
          f"{args.tech_per_job * args.inits_per_tech} trials, n={n}")
    print(f"  output: {args.output}")

    done = existing_keys(args.output)
    write_header = not os.path.exists(args.output)
    f_out = open(args.output, 'a', newline='')
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    if write_header:
        writer.writeheader()

    n_done = 0
    n_unconv = 0
    n_cycled = 0
    for tech_idx in range(args.tech_per_job):
        tech_seed = (args.base_seed * 10_000 + tech_idx) % SEED_MOD
        rng_tech = np.random.default_rng(tech_seed)
        a_arr = np.full(n, A_VALUE)
        b_arr = _draw_b(point_cfg, n, rng_tech)
        z_arr = _draw_z(point_cfg, n, rng_tech)

        base_ns = generate_base_network(
            n=n, c=C, cc=CC,
            aisi_spread=point_cfg['aisi'],
            seed=int(tech_seed),
            a=a_arr, b=b_arr,
            sigma_w=point_cfg['sigma_w'],
        )

        for init_idx in range(args.inits_per_tech):
            init_seed = (tech_seed * 100 + init_idx) % SEED_MOD
            if (int(tech_seed), int(init_seed)) in done:
                continue

            init_ns = generate_random_initial_network(
                n=n, Wbar=base_ns['Wbar'], AiSi=base_ns['AiSi'],
                seed=int(init_seed),
            )
            result = run_unified_simulation(
                init_ns, a_arr, b_arr, z_arr,
                mode='full',
                seed=int((init_seed + 31337) % SEED_MOD),
                max_swaps=kappa,
                nb_rounds=args.nb_rounds,
                chi=args.chi,
            )

            # Static-gap (theta) query at the terminal state.
            p_current, p_best, _, theta_static = compute_static_gap(
                base_ns, result['final_supplier_list'],
                a_arr, b_arr, z_arr, static_max_swaps,
            )

            converged = bool(result['converged'])
            cycle_p = result.get('cycle_period')
            cycle_p_int = int(cycle_p) if isinstance(cycle_p, int) else None
            if not converged:
                n_unconv += 1
            if cycle_p_int is not None and cycle_p_int >= 2:
                n_cycled += 1

            sum_p_hist = result['sum_p_history']

            writer.writerow({
                'point':             args.point,
                'chi':               args.chi,
                'tech_seed':         int(tech_seed),
                'init_seed':         int(init_seed),
                'kappa':             int(kappa),
                'converged':         int(converged),
                'cycle_period':      cycle_p_int if cycle_p_int is not None else '',
                'rounds':            int(result['rounds']),
                'total_rewirings':   int(result['total_rewirings']),
                'sum_p_init':        float(sum_p_hist[0]),
                'sum_p_final_last':  float(sum_p_hist[-1]),
                'sum_p_current':     float(p_current.sum()),
                'sum_p_static_best': float(p_best.sum()),
                'theta_static':      float(theta_static),
                'final_config_hash': _config_hash(result['final_supplier_list']),
            })
            n_done += 1
            if n_done % 25 == 0:
                f_out.flush()
                print(f"  ... {n_done} trials done "
                      f"({n_unconv} non-converged, {n_cycled} cycled)")

    f_out.close()
    print(f"Done. {n_done} trials written to {args.output}  "
          f"(non-converged: {n_unconv}, cycled: {n_cycled}).")


if __name__ == '__main__':
    main()
