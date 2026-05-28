"""Plot the (single-shock, 3-phase) shock experiment.

Reads pickle file(s) written by `shock_experiment.py` from THIS SCRIPT'S
DIRECTORY and writes one PNG per pickle to the same directory.

Layout (two panels, no suptitle):
    Left  (a): 2D MDS embedding of the network-distance matrix.
    Right (b): distance d(M_t, M_baseline) over the global swap index t.

`M_baseline` defaults to the initial random configuration (M_0). Pass
`--baseline final` to plot distance from M_3 instead.

Usage (from anywhere):
    python results/shock/plot_shock.py                 # processes every
                                                       # shock_*.pkl
                                                       # (excluding loops)
    python results/shock/plot_shock.py path/to/x.pkl   # one file
    python results/shock/plot_shock.py --baseline final
"""
import argparse
import glob
import math
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Tracked code in campaigns/shock/; pickles + output PNGs live in results/shock/.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, 'results', 'shock')


PHASE_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}  # blue / orange / green
PHASE_LABELS = {
    1: 'Phase 1: initial wave',
    2: 'Phase 2: one firm randomly removed',
    3: 'Phase 3: same firm re-entered',
}


# Font scaling: figure is 13 in wide; embedded at A4 \textwidth (~6.27 in
# with 1-in margins) it scales by ~0.48. Target on-page sizes (body 8 pt,
# ticks 6.5 pt) are standard for full-width manuscript figures.
FONT_SCALE = 2.1
PLOT_RCPARAMS = {
    'font.size':             round( 8   * FONT_SCALE),   # 17
    'axes.titlesize':        round( 9   * FONT_SCALE),   # 19
    'axes.labelsize':        round( 8   * FONT_SCALE),   # 17
    'xtick.labelsize':       round( 6.5 * FONT_SCALE),   # 14
    'ytick.labelsize':       round( 6.5 * FONT_SCALE),   # 14
    'legend.fontsize':       round( 6.5 * FONT_SCALE),   # 14
    'legend.title_fontsize': round( 7   * FONT_SCALE),   # 15
    'lines.linewidth':       2.0,
    'lines.markersize':      8,
    'axes.linewidth':        1.2,
    'xtick.major.width':     1.2,
    'ytick.major.width':     1.2,
}


def edges_to_set(arr):
    """Convert a (K, 2) edge array into a frozenset of (supplier, buyer) tuples."""
    return frozenset((int(s), int(b)) for s, b in arr)


def cosine_edge_distance(e1, e2):
    """1 - Ochiai (cosine) distance on binary edge sets, in [0, 1].
    Matches the manuscript's definition
        d(M_s, M_t) = 1 - <M_s, M_t> / sqrt(<M_s,M_s> * <M_t,M_t>)
    on the binary adjacency representation, since both inner products equal
    the edge counts of the corresponding configuration.
    """
    if not e1 or not e2:
        return 1.0
    return 1.0 - len(e1 & e2) / math.sqrt(len(e1) * len(e2))


def collect_visits(result):
    """Walk the three phases' swap_edges traces.

    Returns edges_list, phase_arr, t_arr, stable_idx (idx of M_k for k=0..3).
    """
    edges_list = []
    phase_list = []
    t_list = []
    stable_idx = {}

    t_offset = 0
    for phase_num, key in [(1, 'phase1'), (2, 'phase2'), (3, 'phase3')]:
        tr = result[key]['trace']
        swap_edges = tr['swap_edges']
        steps = tr['price_steps']
        if phase_num == 1:
            stable_idx[0] = len(edges_list)
        for k, (e, t_local) in enumerate(zip(swap_edges, steps)):
            if k > 0 and edges_to_set(e) == edges_to_set(swap_edges[k - 1]):
                continue
            edges_list.append(edges_to_set(e))
            phase_list.append(phase_num)
            t_list.append(t_offset + t_local)
        if steps:
            t_offset += steps[-1]
        stable_idx[phase_num] = len(edges_list) - 1

    return edges_list, np.array(phase_list), np.array(t_list), stable_idx


def pairwise_distance_matrix(edges_list):
    n = len(edges_list)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_edge_distance(edges_list[i], edges_list[j])
            D[i, j] = D[j, i] = d
    return D


def mds_embed(D):
    """Classical MDS to 2D. Returns (n, 2) coordinates and variance explained."""
    n = D.shape[0]
    D2 = D.astype(float) ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:2]
    L = np.diag(np.sqrt(np.maximum(eigvals[idx], 0)))
    coords = eigvecs[:, idx] @ L
    total_var = np.sum(np.maximum(eigvals, 0))
    var_explained = (np.sum(np.maximum(eigvals[idx], 0)) / total_var
                     if total_var > 0 else 0.0)
    return coords, var_explained


def make_figure(result, baseline='initial', save_path=None):
    edges_list, phases, ts, stable_idx = collect_visits(result)
    D = pairwise_distance_matrix(edges_list)
    coords, var_explained = mds_embed(D)

    baseline_idx = stable_idx[0] if baseline == 'initial' else stable_idx[3]
    d_from_baseline = D[baseline_idx, :]

    with plt.rc_context(PLOT_RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.8),
                                 constrained_layout=True)

        # ----------------------------------------------------- (a) TS (left)
        ax_ts = axes[0]
        phase_starts = {p: ts[phases == p].min() for p in (1, 2, 3)}
        phase_ends   = {p: ts[phases == p].max() for p in (1, 2, 3)}
        for p in (1, 2, 3):
            ax_ts.axvspan(phase_starts[p], phase_ends[p],
                          color=PHASE_COLORS[p], alpha=0.08)
        for p in (1, 2, 3):
            mask = phases == p
            ax_ts.plot(ts[mask], d_from_baseline[mask], 'o-',
                       color=PHASE_COLORS[p], ms=5, lw=1.6,
                       label=PHASE_LABELS[p], alpha=0.92)
        for k in (0, 1, 2, 3):
            idx = stable_idx[k]
            ax_ts.scatter(ts[idx], d_from_baseline[idx], marker='*', s=240,
                          facecolor='white', edgecolor='black',
                          linewidth=1.6, zorder=10)
            ax_ts.annotate(f'$M_{k}$', (ts[idx], d_from_baseline[idx]),
                           xytext=(6, 8), textcoords='offset points')
        ax_ts.set_xlabel(r'time step $t$')
        if baseline == 'initial':
            ax_ts.set_ylabel(r'Distance to $M_0$')
        else:
            ax_ts.set_ylabel(r'Distance to $M_3$')
        ax_ts.set_title('(a) Drift over time', loc='left')
        ax_ts.grid(alpha=0.3)
        ax_ts.legend(loc='lower right', framealpha=0.95)

        # ----------------------------------------------------- (b) MDS (right)
        ax_mds = axes[1]
        for phase in (1, 2, 3):
            mask = phases == phase
            ax_mds.plot(coords[mask, 0], coords[mask, 1], '-',
                        color=PHASE_COLORS[phase], alpha=0.4, lw=1.2)
            ax_mds.scatter(coords[mask, 0], coords[mask, 1], s=40,
                           color=PHASE_COLORS[phase], alpha=0.85,
                           edgecolors='none', label=PHASE_LABELS[phase])
        for k, marker, ms in [(0, 'X', 18), (1, '*', 22), (2, '*', 22),
                              (3, '*', 22)]:
            idx = stable_idx[k]
            ax_mds.scatter(coords[idx, 0], coords[idx, 1], marker=marker,
                           s=ms ** 2, facecolor='white', edgecolor='black',
                           linewidth=1.6, zorder=10)
            ax_mds.annotate(f'$M_{k}$', (coords[idx, 0], coords[idx, 1]),
                            xytext=(8, 8), textcoords='offset points')
        ax_mds.set_xlabel('MDS axis 1')
        ax_mds.set_ylabel('MDS axis 2')
        ax_mds.set_title(f'(b) Network space '
                         f'(MDS, {var_explained*100:.0f}% variance)',
                         loc='left')
        ax_mds.grid(alpha=0.3)
        # Legend already on panel (a); none here.

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Wrote {save_path}")
        plt.close(fig)


def print_summary(result):
    edges_list, phases, ts, stable_idx = collect_visits(result)
    D = pairwise_distance_matrix(edges_list)
    print("\n=== Shock-experiment summary (cosine distance) ===")
    for k in (0, 1, 2, 3):
        idx = stable_idx[k]
        print(f"  M{k}:  visited at t={ts[idx]:>4d},  "
              f"d(M{k}, M0) = {D[stable_idx[0], idx]:.4f}")
    print(f"\n  d(M1, M2) = {D[stable_idx[1], stable_idx[2]]:.4f}")
    print(f"  d(M2, M3) = {D[stable_idx[2], stable_idx[3]]:.4f}")
    print(f"  d(M1, M3) = {D[stable_idx[1], stable_idx[3]]:.4f}")
    for p in (1, 2, 3):
        n_visits = (phases == p).sum()
        print(f"  phase {p}: {n_visits} configurations visited "
              f"({result[f'phase{p}']['rounds']} rounds, "
              f"{result[f'phase{p}']['total_rewirings']} rewirings)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', nargs='?', default=None,
                   help=f'Pickle path. Default: every shock_*.pkl in '
                        f'{DEFAULT_DATA_DIR}, excluding shock_loop_*.pkl.')
    p.add_argument('--baseline', choices=['initial', 'final'],
                   default='initial')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if args.input is not None:
        targets = [args.input]
    else:
        all_pkls = sorted(glob.glob(os.path.join(DEFAULT_DATA_DIR, 'shock_*.pkl')))
        targets = [p for p in all_pkls
                   if not os.path.basename(p).startswith('shock_loop_')]
        if not targets:
            raise FileNotFoundError(
                f"No shock_*.pkl files in {DEFAULT_DATA_DIR} "
                f"(after excluding shock_loop_*.pkl)."
            )

    for path in targets:
        with open(path, 'rb') as f:
            result = pickle.load(f)
        print(f"\n--- {os.path.basename(path)} ---")
        print_summary(result)
        save_path = os.path.splitext(path)[0] + '.png'
        make_figure(result, baseline=args.baseline, save_path=save_path)


if __name__ == '__main__':
    main()
