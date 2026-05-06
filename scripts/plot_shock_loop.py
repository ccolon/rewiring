"""Plot the K-cycle shock-loop experiment.

Reads a pickle written by `shock_loop_experiment.py` and produces a two-panel
figure showing cumulative drift across the K cycles:

    Left  (MDS): 2D MDS embedding of all visited configurations, colored by
                 cycle index k via a sequential colormap. The initial stable
                 config M_0 is marked, plus each cycle's M_shut(k) and
                 M_full(k) endpoints.
    Right (TS):  distance d(M_t, M_0_stable) over the global swap index t,
                 with cycle shading (alternating bands). Shows the cumulative
                 drift of the system away from the first stable config.

Usage:
    python scripts/plot_shock_loop.py results/shock/shock_loop_K10_seed0.pkl
"""
import argparse
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def edges_to_set(arr):
    return frozenset((int(s), int(b)) for s, b in arr)


def hamming_edge_distance(e1, e2):
    return len(e1 ^ e2)


def collect_all_visits(result):
    """Walk all phases (init + K cycles of shutdown+reintroduction).

    Returns:
        edges_list  : list of frozenset edge-sets
        cycle_arr   : np.array of cycle id (0 = init phase, 1..K = cycles)
        sub_arr     : np.array of sub-phase ('init', 'shut', 'full')
        t_arr       : cumulative global swap index
        markers     : dict {('init', 0): idx of M_init_stable,
                            ('shut', k): idx of M_shut(k),
                            ('full', k): idx of M_full(k) for k=1..K}
    """
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

    # Init phase (cycle 0)
    append_phase(result['phase_init']['trace'], 0, 'init')

    # K cycles
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
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_edge_distance(edges_list[i], edges_list[j])
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
    var_explained = np.sum(np.maximum(eigvals[idx], 0)) / total_var if total_var > 0 else 0.0
    return coords, var_explained


def make_figure(result, save_path=None):
    edges_list, cycles, subs, ts, markers = collect_all_visits(result)
    D = pairwise_distance_matrix(edges_list)
    coords, var_explained = mds_embed(D)

    K = result['config']['n_shocks']
    cmap = plt.get_cmap('viridis')
    cycle_color = lambda k: cmap(k / max(K, 1))

    # Distance from M_init_stable (the post-phase-0 fixed point)
    baseline_idx = markers[('init', 0)]
    d_from_baseline = D[baseline_idx, :]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), constrained_layout=True)

    # =========================================================================
    # Panel A: MDS embedding
    # =========================================================================
    axA = axes[0]
    # Init-phase trajectory (gray, faded)
    mask = (cycles == 0)
    axA.plot(coords[mask, 0], coords[mask, 1], '-', color='gray', alpha=0.4, lw=1.0)
    axA.scatter(coords[mask, 0], coords[mask, 1], s=14, color='gray', alpha=0.6,
                edgecolors='none', label='init phase')
    # Each cycle: shutdown + reintroduction, single color
    for k in range(1, K + 1):
        mask = (cycles == k)
        col = cycle_color(k)
        axA.plot(coords[mask, 0], coords[mask, 1], '-', color=col, alpha=0.5, lw=1.0)
        axA.scatter(coords[mask, 0], coords[mask, 1], s=18, color=col, alpha=0.85,
                    edgecolors='none')
    # Mark M_init_stable
    idx = markers[('init', 0)]
    axA.scatter(coords[idx, 0], coords[idx, 1], marker='X', s=180,
                facecolor='white', edgecolor='black', linewidth=1.5, zorder=10)
    axA.annotate('$M_0$', (coords[idx, 0], coords[idx, 1]),
                 xytext=(8, 8), textcoords='offset points', fontsize=11)
    # Mark every reintroduction stable point (M_full), sparingly labeled
    for k in range(1, K + 1):
        idx = markers[('full', k)]
        axA.scatter(coords[idx, 0], coords[idx, 1], marker='*', s=120,
                    facecolor=cycle_color(k), edgecolor='black', linewidth=0.8, zorder=9)
        if k in (1, K // 2, K):  # label first, middle, last for clarity
            axA.annotate(f'$M_{{{k}}}$', (coords[idx, 0], coords[idx, 1]),
                         xytext=(6, 6), textcoords='offset points', fontsize=9)
    axA.set_xlabel('MDS axis 1')
    axA.set_ylabel('MDS axis 2')
    axA.set_title(f'{K}-cycle trajectory in network space '
                  f'(MDS, {var_explained*100:.0f}% variance)')
    axA.grid(alpha=0.3)
    # Colorbar for cycle index
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=1, vmax=K))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axA, fraction=0.04, pad=0.02)
    cbar.set_label('cycle index $k$')

    # =========================================================================
    # Panel B: distance-from-M_init time series with cycle shading
    # =========================================================================
    axB = axes[1]
    # Determine cycle boundaries (start/end t for each cycle)
    cycle_t_starts = {}
    cycle_t_ends = {}
    for k_id in np.unique(cycles):
        mask = cycles == k_id
        cycle_t_starts[k_id] = ts[mask].min()
        cycle_t_ends[k_id] = ts[mask].max()
    # Shade alternating cycles
    for k in range(1, K + 1):
        col = cycle_color(k)
        axB.axvspan(cycle_t_starts[k], cycle_t_ends[k], color=col, alpha=0.10)
    # Plot init phase line in gray
    mask = cycles == 0
    axB.plot(ts[mask], d_from_baseline[mask], '-', color='gray', lw=1.0, alpha=0.7,
             label='init phase')
    # Plot each cycle's points
    for k in range(1, K + 1):
        mask = cycles == k
        col = cycle_color(k)
        axB.plot(ts[mask], d_from_baseline[mask], 'o-', color=col, ms=2.5, lw=1.0,
                 alpha=0.85)
    # Mark stable configs
    idx = markers[('init', 0)]
    axB.scatter(ts[idx], d_from_baseline[idx], marker='X', s=180,
                facecolor='white', edgecolor='black', linewidth=1.5, zorder=10)
    for k in range(1, K + 1):
        idx = markers[('full', k)]
        axB.scatter(ts[idx], d_from_baseline[idx], marker='*', s=120,
                    facecolor=cycle_color(k), edgecolor='black', linewidth=0.8, zorder=10)
    axB.set_xlabel('global swap index $t$')
    axB.set_ylabel('Hamming distance to $M_0$ (first stable config)')
    axB.set_title(f'Cumulative drift over {K} shock cycles')
    axB.grid(alpha=0.3)

    cfg = result['config']
    fig.suptitle(
        f"K-cycle shock experiment: n={cfg['n']}, c={cfg['c']}, cc={cfg['cc']}, "
        f"ms={cfg['max_swaps']}, mode={cfg['mode']}, "
        f"a={cfg['a_config']}, b={cfg['b_config']}, z={cfg['z_config']}, "
        f"aisi={cfg['aisi_spread']}, sw={cfg['sigma_w']}",
        fontsize=10
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Wrote {save_path}")
    return fig


def print_summary(result):
    edges_list, cycles, subs, ts, markers = collect_all_visits(result)
    D = pairwise_distance_matrix(edges_list)
    K = result['config']['n_shocks']
    base = markers[('init', 0)]
    print("\n=== K-cycle shock summary ===")
    print(f"  M_0 (init stable):   d(M_0, M_0) = 0")
    for k in range(1, K + 1):
        idx_shut = markers[('shut', k)]
        idx_full = markers[('full', k)]
        d_shut = D[base, idx_shut]
        d_full = D[base, idx_full]
        prev_full = markers[('full', k - 1)] if k > 1 else markers[('init', 0)]
        d_prev = D[prev_full, idx_full]
        print(f"  cycle {k:>2d}: firm={result['cycles'][k-1]['firm_idx']:>3d}  "
              f"d(M_shut, M_0)={d_shut:>3d}  "
              f"d(M_{k}, M_0)={d_full:>3d}  "
              f"d(M_{k}, M_{k-1})={d_prev:>3d}  "
              f"rewirings(shut/full)="
              f"{result['cycles'][k-1]['shutdown']['total_rewirings']:>3d}/"
              f"{result['cycles'][k-1]['reintroduction']['total_rewirings']:>3d}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input')
    p.add_argument('--output', default=None)
    args = p.parse_args()

    with open(args.input, 'rb') as f:
        result = pickle.load(f)

    print_summary(result)

    save_path = args.output or os.path.splitext(args.input)[0] + '.png'
    make_figure(result, save_path=save_path)


if __name__ == '__main__':
    main()
