"""Plot the K-cycle shock-loop experiment.

Reads pickle file(s) written by `shock_loop_experiment.py` from THIS
SCRIPT'S DIRECTORY and writes one PNG per pickle to the same directory.

Two-panel layout (no suptitle):
    (a) 2D MDS embedding of all visited configurations, colored by cycle
        index k via a sequential colormap.
    (b) Hamming distance d(M_t, M_0) over the global swap index t with
        per-cycle shading.

Usage (from anywhere):
    python results/shock/plot_shock_loop.py
    python results/shock/plot_shock_loop.py path/to/shock_loop_K10_seed0.pkl
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


# Font scaling matches plot_shock.py: figure ~13.5 in wide; included at
# A4 \textwidth with 1-in margins (~6.27 in) scales by ~0.46. Target
# on-page sizes are 8 pt body / 6.5 pt ticks.
FONT_SCALE = 2.1
PLOT_RCPARAMS = {
    'font.size':             round( 8   * FONT_SCALE),
    'axes.titlesize':        round( 9   * FONT_SCALE),
    'axes.labelsize':        round( 8   * FONT_SCALE),
    'xtick.labelsize':       round( 6.5 * FONT_SCALE),
    'ytick.labelsize':       round( 6.5 * FONT_SCALE),
    'legend.fontsize':       round( 6.5 * FONT_SCALE),
    'legend.title_fontsize': round( 7   * FONT_SCALE),
    'lines.linewidth':       2.0,
    'lines.markersize':      8,
    'axes.linewidth':        1.2,
    'xtick.major.width':     1.2,
    'ytick.major.width':     1.2,
}


def edges_to_set(arr):
    return frozenset((int(s), int(b)) for s, b in arr)


def cosine_edge_distance(e1, e2):
    """1 - Ochiai (cosine) distance on binary edge sets, in [0, 1]."""
    if not e1 or not e2:
        return 1.0
    return 1.0 - len(e1 & e2) / math.sqrt(len(e1) * len(e2))


def collect_all_visits(result):
    """Walk all phases (init + K cycles of shutdown+reintroduction)."""
    edges_list = []
    cycle_list = []
    sub_list = []
    t_list = []
    markers = {}

    t_offset = 0

    def append_phase(tr, cycle_id, sub):
        nonlocal t_offset
        for k, (e, t_local) in enumerate(zip(tr['swap_edges'], tr['price_steps'])):
            if k > 0 and edges_to_set(e) == edges_to_set(tr['swap_edges'][k - 1]):
                continue
            edges_list.append(edges_to_set(e))
            cycle_list.append(cycle_id)
            sub_list.append(sub)
            t_list.append(t_offset + t_local)
        if tr['price_steps']:
            t_offset += tr['price_steps'][-1]
        markers[(sub, cycle_id)] = len(edges_list) - 1

    append_phase(result['phase_init']['trace'], 0, 'init')
    for cyc in result['cycles']:
        append_phase(cyc['shutdown']['trace'], cyc['k'], 'shut')
        append_phase(cyc['reintroduction']['trace'], cyc['k'], 'full')

    return (edges_list,
            np.array(cycle_list),
            np.array(sub_list, dtype=object),
            np.array(t_list),
            markers)


def pairwise_distance_matrix(edges_list):
    n = len(edges_list)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_edge_distance(edges_list[i], edges_list[j])
            D[i, j] = D[j, i] = d
    return D


def mds_embed(D):
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


def make_figure(result, save_path=None):
    edges_list, cycles, subs, ts, markers = collect_all_visits(result)
    D = pairwise_distance_matrix(edges_list)
    coords, var_explained = mds_embed(D)

    K = result['config']['n_shocks']
    cmap = plt.get_cmap('viridis')
    cycle_color = lambda k: cmap(k / max(K, 1))

    baseline_idx = markers[('init', 0)]
    d_from_baseline = D[baseline_idx, :]

    with plt.rc_context(PLOT_RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8),
                                 constrained_layout=True)

        # ----------------------------------------------------- (a) TS (left)
        ax_ts = axes[0]
        cycle_t_starts = {}
        cycle_t_ends = {}
        for k_id in np.unique(cycles):
            mask = cycles == k_id
            cycle_t_starts[k_id] = ts[mask].min()
            cycle_t_ends[k_id] = ts[mask].max()
        for k in range(1, K + 1):
            col = cycle_color(k)
            ax_ts.axvspan(cycle_t_starts[k], cycle_t_ends[k], color=col,
                          alpha=0.10)
        mask = cycles == 0
        ax_ts.plot(ts[mask], d_from_baseline[mask], '-', color='gray',
                   lw=1.4, alpha=0.8, label='Init phase')
        for k in range(1, K + 1):
            mask = cycles == k
            col = cycle_color(k)
            ax_ts.plot(ts[mask], d_from_baseline[mask], 'o-', color=col,
                       ms=3.5, lw=1.2, alpha=0.85)
        idx = markers[('init', 0)]
        ax_ts.scatter(ts[idx], d_from_baseline[idx], marker='X', s=240,
                      facecolor='white', edgecolor='black', linewidth=1.6,
                      zorder=10)
        for k in range(1, K + 1):
            idx = markers[('full', k)]
            ax_ts.scatter(ts[idx], d_from_baseline[idx], marker='*', s=180,
                          facecolor=cycle_color(k), edgecolor='black',
                          linewidth=0.9, zorder=10)
        ax_ts.set_xlabel(r'time step $t$')
        ax_ts.set_ylabel(r'Distance to $M_0$')
        ax_ts.set_title('(a) Drift over time', loc='left')
        ax_ts.grid(alpha=0.3)
        ax_ts.legend(loc='lower right', framealpha=0.95)

        # ----------------------------------------------------- (b) MDS (right)
        ax_mds = axes[1]
        mask = (cycles == 0)
        ax_mds.plot(coords[mask, 0], coords[mask, 1], '-', color='gray',
                    alpha=0.4, lw=1.0)
        ax_mds.scatter(coords[mask, 0], coords[mask, 1], s=22, color='gray',
                       alpha=0.6, edgecolors='none')
        for k in range(1, K + 1):
            mask = (cycles == k)
            col = cycle_color(k)
            ax_mds.plot(coords[mask, 0], coords[mask, 1], '-', color=col,
                        alpha=0.5, lw=1.0)
            ax_mds.scatter(coords[mask, 0], coords[mask, 1], s=26, color=col,
                           alpha=0.85, edgecolors='none')
        idx = markers[('init', 0)]
        ax_mds.scatter(coords[idx, 0], coords[idx, 1], marker='X', s=240,
                       facecolor='white', edgecolor='black', linewidth=1.6,
                       zorder=10)
        ax_mds.annotate(r'$M_0$', (coords[idx, 0], coords[idx, 1]),
                        xytext=(8, 8), textcoords='offset points')
        for k in range(1, K + 1):
            idx = markers[('full', k)]
            ax_mds.scatter(coords[idx, 0], coords[idx, 1], marker='*', s=180,
                           facecolor=cycle_color(k), edgecolor='black',
                           linewidth=0.9, zorder=9)
            if k in (1, K // 2, K):
                ax_mds.annotate(fr'$M_{{{k}}}$',
                                (coords[idx, 0], coords[idx, 1]),
                                xytext=(6, 6), textcoords='offset points')
        ax_mds.set_xlabel('MDS axis 1')
        ax_mds.set_ylabel('MDS axis 2')
        ax_mds.set_title(f'(b) Network space '
                         f'(MDS, {var_explained*100:.0f}% variance)',
                         loc='left')
        ax_mds.grid(alpha=0.3)
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(vmin=1, vmax=K))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax_mds, fraction=0.045, pad=0.02)
        cbar.set_label(r'cycle index $k$')

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Wrote {save_path}")
        plt.close(fig)


def print_summary(result):
    edges_list, cycles, subs, ts, markers = collect_all_visits(result)
    D = pairwise_distance_matrix(edges_list)
    K = result['config']['n_shocks']
    base = markers[('init', 0)]
    print("\n=== K-cycle shock summary (cosine distance) ===")
    print(f"  M_0 (init stable):   d(M_0, M_0) = 0.0000")
    for k in range(1, K + 1):
        idx_shut = markers[('shut', k)]
        idx_full = markers[('full', k)]
        d_shut = D[base, idx_shut]
        d_full = D[base, idx_full]
        prev_full = markers[('full', k - 1)] if k > 1 else markers[('init', 0)]
        d_prev = D[prev_full, idx_full]
        print(f"  cycle {k:>2d}: firm={result['cycles'][k-1]['firm_idx']:>3d}  "
              f"d(M_shut, M_0)={d_shut:.4f}  "
              f"d(M_{k}, M_0)={d_full:.4f}  "
              f"d(M_{k}, M_{k-1})={d_prev:.4f}  "
              f"rewirings(shut/full)="
              f"{result['cycles'][k-1]['shutdown']['total_rewirings']:>3d}/"
              f"{result['cycles'][k-1]['reintroduction']['total_rewirings']:>3d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', nargs='?', default=None,
                   help=f'Pickle path. Default: every shock_loop_*.pkl in '
                        f'{DEFAULT_DATA_DIR}.')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if args.input is not None:
        targets = [args.input]
    else:
        targets = sorted(glob.glob(
            os.path.join(DEFAULT_DATA_DIR, 'shock_loop_*.pkl')))
        if not targets:
            raise FileNotFoundError(
                f"No shock_loop_*.pkl files in {DEFAULT_DATA_DIR}.")

    for path in targets:
        with open(path, 'rb') as f:
            result = pickle.load(f)
        print(f"\n--- {os.path.basename(path)} ---")
        print_summary(result)
        save_path = os.path.splitext(path)[0] + '.png'
        make_figure(result, save_path=save_path)


if __name__ == '__main__':
    main()
