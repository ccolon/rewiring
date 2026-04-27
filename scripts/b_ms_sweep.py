"""
Sweep driver for the (b, max_swaps) small-multiples figure.

Runs `run_unified_simulation` in mode='aa' across 8 cells
(4 b-regimes x 2 max_swaps settings) and S seeds, dumping per-run traces
for later aggregation by b_ms_sweep_analyze.py.

Output layout:
    sweeps/b_ms/<cell>/seed=<s>/
        ts.csv      - per-round scalars
        nets.npz    - compressed edge snapshots (edges_0..edges_T)
        meta.json   - run parameters and outcome summary

Resume-safe: skips (cell, seed) pairs whose three output files already exist.
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import numpy as np
import pandas as pd

# Allow `python scripts/b_ms_sweep.py` from the repo root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from rewiring.networks import generate_base_network, generate_random_initial_network
from rewiring.parameters import generate_a_parameter, generate_parameter
from rewiring.simulation import run_unified_simulation


# =============================================================================
# DEFAULTS (match plan: N=20, cc=2, AiSi_spread=0.1, a=0.5, z=1.0 homogeneous)
# =============================================================================
SWEEP_NAME = "b_ms"
N = 100
C = 4              # base connectivity (== initial suppliers per firm, approx.)
CC = 4             # alternates per firm
AISI_SPREAD = 0.1
A_VALUE = 0.5
Z_VALUE = 1.0
SIGMA_W = 0.0
MODE = "aa"
S_DEFAULT = 30
T_MAX_DEFAULT = 50

# A single cell-level seed fixes Wbar, AiSi, a, b, z per cell so that all trial
# seeds within a cell share the same underlying economy. Only the initial
# network (supplier sets) and the permutation order vary across trial seeds.
CELL_SEED = 42


# =============================================================================
# CELLS
# =============================================================================
def build_cells(cc):
    """Return list of dicts describing the 8 sweep cells.

    Each cell has a name, a `b_config` (passed to generate_parameter),
    and a concrete max_swaps integer.
    """
    ms_low, ms_full = 2, cc
    b_regimes = [
        ("DRS", {"mode": "homogeneous", "value": 0.9}),
        ("CRS", {"mode": "homogeneous", "value": 1.0}),
        ("IRS", {"mode": "homogeneous", "value": 1.1}),
        ("MIX", {"mode": "uniform", "min": 0.9, "max": 1.1}),
    ]
    cells = []
    for ms_label, ms in [("low", ms_low), ("full", ms_full)]:
        for b_label, b_config in b_regimes:
            cells.append({
                "name": f"{b_label}_{ms_label}",
                "b_config": b_config,
                "max_swaps": ms,
            })
    return cells


# =============================================================================
# RUN A SINGLE (cell, seed)
# =============================================================================
def build_cell_economy(cell, N, C, cc):
    """Build the shared-per-cell economy: (a, b, z, base_state).

    Deterministic given cell['b_config'] and CELL_SEED. Same across all trial
    seeds within this cell so that cross-seed comparisons are meaningful.
    """
    random.seed(CELL_SEED)
    np.random.seed(CELL_SEED)
    b = generate_parameter(cell["b_config"], N, "b", verbose=False)
    a = generate_a_parameter({"mode": "homogeneous", "value": A_VALUE}, b, N, verbose=False)
    z = generate_parameter({"mode": "homogeneous", "value": Z_VALUE}, N, "z", verbose=False)
    base_state = generate_base_network(N, C, cc, AISI_SPREAD, seed=CELL_SEED,
                                       a=a, b=b, sigma_w=SIGMA_W)
    return a, b, z, base_state


def run_one(out_root, cell, seed, N, C, cc, T_max, econ=None, verbose=False):
    """Build init state and run the simulation for a (cell, seed) pair.

    Within a cell, Wbar and AiSi are SHARED across all seeds (via `econ`,
    cached per cell). Only the initial supplier sets and the permutation
    order vary across seeds -- this is the 'different_start_same_parameter'
    setup, so cross-seed distance meaningfully probes path dependence.

    Writes ts.csv, nets.npz, meta.json to <out_root>/<cell.name>/seed=<seed>/.
    """
    cell_dir = os.path.join(out_root, cell["name"], f"seed={seed}")
    os.makedirs(cell_dir, exist_ok=True)
    ts_path = os.path.join(cell_dir, "ts.csv")
    nets_path = os.path.join(cell_dir, "nets.npz")
    meta_path = os.path.join(cell_dir, "meta.json")

    if all(os.path.exists(p) for p in (ts_path, nets_path, meta_path)):
        return {"cell": cell["name"], "seed": seed, "skipped": True}

    if econ is None:
        econ = build_cell_economy(cell, N, C, cc)
    a, b, z, base_state = econ

    # Trial-specific initial network: same Wbar/AiSi as the cell economy, but
    # a random initial supplier pick. Use seed offset to avoid collision
    # with CELL_SEED.
    init_state = generate_random_initial_network(
        N, base_state["Wbar"], base_state["AiSi"], seed=10_000 + seed,
    )

    # Trial-specific permutation seed (different stream from init draw)
    perm_seed = 20_000 + seed
    result = run_unified_simulation(
        init_state, a, b, z,
        mode=MODE, seed=perm_seed, max_swaps=cell["max_swaps"],
        nb_rounds=T_max, trace=True,
    )

    trace = result["trace"]
    # Persist scalars
    pd.DataFrame(trace["scalars"]).to_csv(ts_path, index=False)
    # Persist edge snapshots (stack into a single 3D array if all same shape, else dict)
    edges = trace["edges"]
    # All snapshots share the same edge count (rewiring is swap-not-add), so stack
    edges_stack = np.stack(edges, axis=0)  # (T+1, K, 2) int32
    np.savez_compressed(nets_path, edges=edges_stack)
    # Meta
    meta = {
        "sweep_name": SWEEP_NAME,
        "cell": cell["name"],
        "seed": int(seed),
        "cell_seed": int(CELL_SEED),
        "init_seed": int(10_000 + seed),
        "perm_seed": int(perm_seed),
        "N": int(N),
        "c": int(C),
        "cc": int(cc),
        "AiSi_spread": float(AISI_SPREAD),
        "a_value": float(A_VALUE),
        "z_value": float(Z_VALUE),
        "sigma_w": float(SIGMA_W),
        "mode": MODE,
        "max_swaps": int(cell["max_swaps"]),
        "T_max": int(T_max),
        "b_config": cell["b_config"],
        "b_realized_mean": float(np.mean(b)),
        "b_realized_std": float(np.std(b)),
        "converged": bool(result["converged"]),
        "converged_at": trace["converged_at"],
        "rounds_run": int(result["rounds"]),
        "initial_utility": float(result["initial_utility"]),
        "final_utility": float(result["final_utility"]),
        "edge_count": int(edges_stack.shape[1]),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)

    if verbose:
        status = "CONV" if meta["converged"] else "NC"
        print(f"  [{cell['name']} seed={seed}] {status} rounds={meta['rounds_run']} "
              f"final_util={meta['final_utility']:.4f}")
    return {"cell": cell["name"], "seed": seed, "skipped": False, **meta}


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default=f"sweeps/{SWEEP_NAME}",
                        help="Output root directory (default sweeps/b_ms)")
    parser.add_argument("--seeds", type=int, default=S_DEFAULT,
                        help=f"Number of seeds (default {S_DEFAULT})")
    parser.add_argument("--t-max", type=int, default=T_MAX_DEFAULT,
                        help=f"Max rounds per run (default {T_MAX_DEFAULT})")
    parser.add_argument("--n-firms", type=int, default=N,
                        help=f"Number of firms (default {N})")
    parser.add_argument("--cc", type=int, default=CC,
                        help=f"Alternate suppliers per firm (default {CC})")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (1 = sequential)")
    parser.add_argument("--cells", default=None,
                        help="Comma-separated list of cell names to run (default: all 8)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    all_cells = build_cells(args.cc)
    if args.cells is not None:
        keep = set(args.cells.split(","))
        all_cells = [c for c in all_cells if c["name"] in keep]
    print(f"Sweep '{SWEEP_NAME}': {len(all_cells)} cells x {args.seeds} seeds = "
          f"{len(all_cells) * args.seeds} runs -> {args.out_root}")
    for c in all_cells:
        print(f"  cell {c['name']:10s}  b_config={c['b_config']}  max_swaps={c['max_swaps']}")

    # Also dump top-level sweep config for reproducibility
    sweep_cfg = {
        "sweep_name": SWEEP_NAME,
        "N": args.n_firms, "c": C, "cc": args.cc,
        "AiSi_spread": AISI_SPREAD,
        "a_value": A_VALUE, "z_value": Z_VALUE, "sigma_w": SIGMA_W,
        "mode": MODE,
        "seeds": args.seeds, "t_max": args.t_max,
        "cells": all_cells,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    with open(os.path.join(args.out_root, "sweep_config.json"), "w") as fh:
        json.dump(sweep_cfg, fh, indent=2, default=str)

    # Build each cell's economy ONCE and cache (sequential path only -- in
    # parallel workers rebuild their own copies from CELL_SEED, which is fine
    # since the cell-level draws are deterministic).
    econ_by_cell = {c["name"]: build_cell_economy(c, args.n_firms, C, args.cc)
                    for c in all_cells}

    jobs = [(c, s) for c in all_cells for s in range(args.seeds)]
    t0 = datetime.now()

    if args.workers <= 1:
        for idx, (cell, seed) in enumerate(jobs, 1):
            run_one(args.out_root, cell, seed, args.n_firms, C, args.cc,
                    args.t_max, econ=econ_by_cell[cell["name"]],
                    verbose=args.verbose)
            if idx % max(1, len(jobs) // 20) == 0:
                elapsed = (datetime.now() - t0).total_seconds()
                print(f"[{idx}/{len(jobs)}] {elapsed:.1f}s")
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {
                ex.submit(run_one, args.out_root, c, s, args.n_firms, C, args.cc,
                          args.t_max, None, args.verbose): (c["name"], s)
                for c, s in jobs
            }
            done = 0
            for fut in as_completed(futures):
                done += 1
                try:
                    fut.result()
                except Exception as e:
                    name, s = futures[fut]
                    print(f"  FAIL [{name} seed={s}]: {e}", file=sys.stderr)
                if done % max(1, len(jobs) // 20) == 0:
                    elapsed = (datetime.now() - t0).total_seconds()
                    print(f"[{done}/{len(jobs)}] {elapsed:.1f}s")

    elapsed = (datetime.now() - t0).total_seconds()
    print(f"Done: {len(jobs)} runs in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
