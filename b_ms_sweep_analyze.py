"""
Aggregate a (b, max_swaps) sweep and render the 2x4 small-multiples figure.

Reads:
    sweeps/<sweep_name>/<cell>/seed=<s>/{ts.csv,nets.npz,meta.json}

Produces:
    sweeps/<sweep_name>/aggregates.parquet.pkl  (cached per-t distances and churn)
    figures/<sweep_name>_small_multiples.{pdf,png}

Panels:
    rows: max_swaps in {low = cc-1, full = cc}
    cols: b in {DRS 0.9, CRS 1.0, IRS 1.1, MIX U[0.9,1.1]}

Per panel, twin axes:
    left  (log, solid):   log10(rewirings+1) per seed (alpha=0.25) + bold median
    right (linear, dashed): cross-seed distance d_bar_t with 10-90 pct band
"""
import argparse
import json
import os
import pickle
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROW_ORDER = ["low", "full"]
COL_ORDER = ["DRS", "CRS", "IRS", "MIX"]
COL_TITLES = {"DRS": "DRS (b=0.9)", "CRS": "CRS (b=1.0)",
              "IRS": "IRS (b=1.1)", "MIX": "Mixed b~U[0.9,1.1]"}


# =============================================================================
# LOADING
# =============================================================================
def load_sweep(sweep_dir):
    """Walk sweep_dir and return {cell_name: {seed: {'ts', 'edges', 'meta'}}}."""
    runs = {}
    cell_dirs = [d for d in glob(os.path.join(sweep_dir, "*")) if os.path.isdir(d)]
    for cdir in cell_dirs:
        cname = os.path.basename(cdir)
        seed_dirs = glob(os.path.join(cdir, "seed=*"))
        if not seed_dirs:
            continue
        runs[cname] = {}
        for sdir in seed_dirs:
            s = int(os.path.basename(sdir).split("=")[1])
            try:
                ts = pd.read_csv(os.path.join(sdir, "ts.csv"))
                nets = np.load(os.path.join(sdir, "nets.npz"))
                with open(os.path.join(sdir, "meta.json")) as fh:
                    meta = json.load(fh)
            except FileNotFoundError:
                continue
            runs[cname][s] = {"ts": ts, "edges": nets["edges"], "meta": meta}
    return runs


# =============================================================================
# DISTANCE COMPUTATION
# =============================================================================
def edges_to_membership(edges, N):
    """Convert (T+1, K, 2) int32 edges to (T+1, N*N) int8 boolean membership."""
    T1, K, _ = edges.shape
    M = np.zeros((T1, N * N), dtype=np.int8)
    flat = edges[..., 0] * N + edges[..., 1]  # (T1, K)
    rows = np.repeat(np.arange(T1), K)
    cols = flat.ravel()
    M[rows, cols] = 1
    return M


def pad_to_length(arr, T_common, axis=0):
    """Repeat the last row of arr along `axis` so that arr.shape[axis] == T_common."""
    cur = arr.shape[axis]
    if cur >= T_common:
        idx = [slice(None)] * arr.ndim
        idx[axis] = slice(0, T_common)
        return arr[tuple(idx)]
    pad_n = T_common - cur
    idx = [slice(None)] * arr.ndim
    idx[axis] = slice(cur - 1, cur)
    tail = np.repeat(arr[tuple(idx)], pad_n, axis=axis)
    return np.concatenate([arr, tail], axis=axis)


def compute_cell_aggregates(cell_runs, N):
    """Given {seed: {'ts','edges','meta'}} for one cell, return a dict of aggregates.

    Returns:
        t_axis: (T_common,) ints
        churn_per_seed: (S, T_common) ints
        sum_p_per_seed: (S, T_common) floats
        d_bar:  (T_common,) mean pairwise distance
        d_p10:  (T_common,)
        d_p90:  (T_common,)
        self_drift_per_seed: (S, T_common) floats  (distance to own t=0)
        converged_at:  list[int or None] per seed
        seeds: list of seed ids
    """
    seeds = sorted(cell_runs.keys())
    S = len(seeds)
    # Determine common round length; every run logs t=0..T_s inclusive
    T_lens = [int(cell_runs[s]["ts"]["t"].max()) for s in seeds]
    T_common = max(T_lens)

    # Scalars (pad with zeros for churn after convergence; hold sum_p at last)
    churn = np.zeros((S, T_common + 1), dtype=np.int32)
    sum_p = np.zeros((S, T_common + 1), dtype=np.float64)
    for i, s in enumerate(seeds):
        ts = cell_runs[s]["ts"]
        t_vals = ts["t"].to_numpy()
        churn[i, t_vals] = ts["rewirings"].to_numpy().astype(np.int32)
        # sum_p: fill known ts, then hold last value past end
        sp = ts["sum_p"].to_numpy()
        sum_p[i, t_vals] = sp
        last_t = int(t_vals.max())
        if last_t < T_common:
            sum_p[i, last_t + 1:] = sp[-1]

    # Network membership tensor: (S, T_common+1, N*N)
    M = np.zeros((S, T_common + 1, N * N), dtype=np.int8)
    K_per_seed = np.zeros(S, dtype=np.int64)
    for i, s in enumerate(seeds):
        edges_s = cell_runs[s]["edges"]  # (T_s+1, K, 2)
        K_per_seed[i] = edges_s.shape[1]
        Ms = edges_to_membership(edges_s, N)  # (T_s+1, N*N)
        Ms = pad_to_length(Ms, T_common + 1, axis=0)
        M[i] = Ms

    # Pairwise distances per t: batched matmul
    # inter[s, u, t] = M[s, t, :] @ M[u, t, :]
    # Loop over t to keep memory bounded
    d_bar = np.zeros(T_common + 1)
    d_p10 = np.zeros(T_common + 1)
    d_p90 = np.zeros(T_common + 1)
    triu_i, triu_j = np.triu_indices(S, k=1)

    denom = np.sqrt(np.outer(K_per_seed, K_per_seed).astype(np.float64))  # (S, S)
    for t in range(T_common + 1):
        Mt = M[:, t, :].astype(np.float64)
        inter = Mt @ Mt.T  # (S, S)
        d = 1.0 - inter / np.maximum(denom, 1.0)
        vals = d[triu_i, triu_j]
        d_bar[t] = vals.mean() if vals.size else 0.0
        d_p10[t] = np.percentile(vals, 10) if vals.size else 0.0
        d_p90[t] = np.percentile(vals, 90) if vals.size else 0.0

    # Self-drift: d(M[s,0,:], M[s,t,:])
    self_drift = np.zeros((S, T_common + 1))
    for i in range(S):
        e0 = M[i, 0, :].astype(np.float64)
        for t in range(T_common + 1):
            et = M[i, t, :].astype(np.float64)
            inter = e0 @ et
            K = float(K_per_seed[i])
            self_drift[i, t] = 1.0 - inter / max(K, 1.0)

    converged_at = [cell_runs[s]["meta"].get("converged_at") for s in seeds]

    return {
        "t_axis": np.arange(T_common + 1),
        "churn_per_seed": churn,
        "sum_p_per_seed": sum_p,
        "d_bar": d_bar, "d_p10": d_p10, "d_p90": d_p90,
        "self_drift_per_seed": self_drift,
        "converged_at": converged_at,
        "seeds": seeds,
        "K_per_seed": K_per_seed,
    }


# =============================================================================
# AGGREGATES CACHE
# =============================================================================
def compute_or_load_aggregates(sweep_dir, force=False):
    cache_path = os.path.join(sweep_dir, "aggregates.pkl")
    if not force and os.path.exists(cache_path):
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)
    # Load sweep config for N
    with open(os.path.join(sweep_dir, "sweep_config.json")) as fh:
        cfg = json.load(fh)
    N = int(cfg["N"])

    runs = load_sweep(sweep_dir)
    cell_aggregates = {}
    for cname, cell_runs in runs.items():
        if not cell_runs:
            continue
        print(f"  aggregating {cname} ({len(cell_runs)} seeds)...")
        cell_aggregates[cname] = compute_cell_aggregates(cell_runs, N)

    bundle = {"cfg": cfg, "cells": cell_aggregates}
    with open(cache_path, "wb") as fh:
        pickle.dump(bundle, fh)
    return bundle


# =============================================================================
# FIGURE
# =============================================================================
def parse_cell_name(name):
    """'DRS_low' -> ('DRS', 'low')."""
    parts = name.rsplit("_", 1)
    return parts[0], parts[1]


def render_figure(bundle, out_path_prefix, left_metric="rewirings"):
    """Render the 2x4 small-multiples figure.

    left_metric:
        'rewirings' -> left axis is churn (rewirings per round)
        'cost'      -> left axis is total cost, sum_i p_i
    """
    cfg = bundle["cfg"]
    cells = bundle["cells"]

    if left_metric == "rewirings":
        left_key = "churn_per_seed"
        left_ylabel = "rewirings per round"
        left_legend = "median rewirings"
        left_seed_legend = "per-seed rewirings"
        share_left = "row"
    elif left_metric == "cost":
        left_key = "sum_p_per_seed"
        left_ylabel = r"total cost $\sum_i p_i$"
        left_legend = "median total cost"
        left_seed_legend = "per-seed total cost"
        share_left = "row"
    else:
        raise ValueError(f"Unknown left_metric: {left_metric!r}")

    fig, axes = plt.subplots(2, 4, sharex=True, sharey=share_left,
                             figsize=(11, 5.0), constrained_layout=True)

    for cname, agg in cells.items():
        b_label, ms_label = parse_cell_name(cname)
        if b_label not in COL_ORDER or ms_label not in ROW_ORDER:
            continue
        row = ROW_ORDER.index(ms_label)
        col = COL_ORDER.index(b_label)
        ax = axes[row, col]
        ax2 = ax.twinx()
        if col != 3:
            ax2.set_yticklabels([])
        ax2.set_ylim(-0.02, 0.52)

        t = agg["t_axis"]
        left = agg[left_key]  # (S, T+1)
        d_bar = agg["d_bar"]; d_p10 = agg["d_p10"]; d_p90 = agg["d_p90"]
        S = left.shape[0]

        # Left: per-seed trajectories + median. Rewirings are only defined for
        # r >= 1 (no round 0), so skip t=0 for that metric. Cost at t=0 is the
        # initial equilibrium total cost and is kept.
        if left_metric == "rewirings":
            t_left = t[1:]
            left_plot = left[:, 1:]
        else:
            t_left = t
            left_plot = left
        for i in range(S):
            ax.plot(t_left, left_plot[i], color="C0", lw=0.4, alpha=0.25)
        med = np.median(left_plot, axis=0)
        ax.plot(t_left, med, color="C0", lw=1.6, label=left_legend)

        # Right: cross-seed distance (defined from t=0 -- initial-network spread)
        ax2.fill_between(t, d_p10, d_p90, color="C3", alpha=0.18)
        ax2.plot(t, d_bar, color="C3", lw=1.6, ls="--",
                 label=r"$\bar{d}_t$ (10-90% band)")

        # Median converged_at — only meaningful if a majority of seeds converged
        conv = [c for c in agg["converged_at"] if c is not None]
        if len(conv) >= 0.5 * S:
            ax.axvline(np.median(conv), color="grey", ls=":", lw=0.8)

        if row == 0:
            ax.set_title(COL_TITLES[b_label], fontsize=10)
        if col == 0:
            ax.set_ylabel(f"max_swaps = c_i{'-1' if ms_label=='low' else ''}\n"
                          f"{left_ylabel}", fontsize=9)
        if row == 1:
            ax.set_xlabel("round t")
        if col == 3:
            ax2.set_ylabel(r"cross-seed distance $\bar{d}_t$", color="C3",
                           fontsize=9)
            ax2.tick_params(axis="y", colors="C3")

    # Shared legend
    handles = [
        plt.Line2D([0], [0], color="C0", lw=1.6, label=left_legend),
        plt.Line2D([0], [0], color="C0", lw=0.5, alpha=0.5, label=left_seed_legend),
        plt.Line2D([0], [0], color="C3", lw=1.6, ls="--", label=r"$\bar{d}_t$"),
        plt.Rectangle((0, 0), 1, 1, fc="C3", alpha=0.18, label="10-90% band"),
        plt.Line2D([0], [0], color="grey", ls=":", lw=0.8,
                   label=r"median convergence round ($\geq 50\%$ seeds)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.03), frameon=False, fontsize=9)

    fig.suptitle(
        f"Small-multiples of AA simulation traces "
        f"(N={cfg['N']}, cc={cfg['cc']}, S={cfg['seeds']} seeds)",
        fontsize=11,
    )

    for ext in ("pdf", "png"):
        out = f"{out_path_prefix}.{ext}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-dir", default="sweeps/b_ms")
    parser.add_argument("--fig-dir", default="figures")
    parser.add_argument("--force", action="store_true",
                        help="Recompute aggregates from raw outputs")
    args = parser.parse_args()

    os.makedirs(args.fig_dir, exist_ok=True)
    print(f"Loading sweep from {args.sweep_dir} ...")
    bundle = compute_or_load_aggregates(args.sweep_dir, force=args.force)
    print(f"Cells: {sorted(bundle['cells'].keys())}")

    sweep_name = os.path.basename(os.path.normpath(args.sweep_dir))
    base = os.path.join(args.fig_dir, f"{sweep_name}_small_multiples")
    render_figure(bundle, base, left_metric="rewirings")
    render_figure(bundle, f"{base}_cost", left_metric="cost")


if __name__ == "__main__":
    main()
