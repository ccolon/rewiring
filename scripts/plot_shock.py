"""Plot the shock experiment.

Reads a pickle written by `shock_experiment.py` and produces a two-panel figure:

    Left  (MDS): 2D MDS embedding of the network-distance matrix, with each
                 visited configuration as a point coloured by phase. Lines
                 connect successive visits. Stable configurations M0, M1, M2,
                 M3 are marked with stars.
    Right (TS):  distance d(M_t, M_baseline) over the global swap index t,
                 with phase shading. The "wave" structure (rapid rise during
                 each phase, then plateau at the next stable config) is the
                 key visual.

`M_baseline` defaults to the initial random configuration (M0) so that the
right panel reads as "drift away from the unstable starting point". Pass
`--baseline final` to instead plot distance from the very last stable config
M3 (i.e., reverse time framing).

Usage:
    python scripts/plot_shock.py results/shock/shock_seed0.pkl
    python scripts/plot_shock.py results/shock/shock_seed0.pkl --baseline final
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

PHASE_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}     # blue / orange / green
PHASE_LABELS = {1: 'Phase 1: random init -> M1',
                2: 'Phase 2: shock (firm out) -> M2',
                3: 'Phase 3: reintroduction -> M3'}


def edges_to_set(arr):
    """Convert the (K, 2) edge array into a frozenset of (supplier, buyer) tuples."""
    return frozenset((int(s), int(b)) for s, b in arr)


def hamming_edge_distance(e1, e2):
    """Symmetric difference cardinality between two edge sets."""
    return len(e1 ^ e2)


def collect_visits(result):
    """Walk the three phases' swap_edges traces and return:

        edges_list  : list of frozenset edge-sets, one per visited config
        phase_arr   : np.array of phase id (1/2/3), one per visit
        t_arr       : np.array of global swap index (cumulative across phases)
        stable_idx  : dict {0: idx of M0, 1: idx of M1, 2: idx of M2, 3: idx of M3}
                      pointing into edges_list/phase_arr/t_arr.
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
        # The first entry is the initial config of this phase (at t=0 of the phase).
        # phase 1's first entry is M0; phases 2 and 3's first entries are M1' and M2.
        if phase_num == 1:
            stable_idx[0] = len(edges_list)             # M0
        for k, (e, t_local) in enumerate(zip(swap_edges, steps)):
            # Skip the "anchor" duplicate at the end if it has the same edge set
            # as the previous one (price-only anchor).
            if k > 0 and edges_to_set(e) == edges_to_set(swap_edges[k - 1]):
                continue
            edges_list.append(edges_to_set(e))
            phase_list.append(phase_num)
            t_list.append(t_offset + t_local)
        # Advance t offset by the last local t in this phase
        if steps:
            t_offset += steps[-1]
        # Mark the stable config at the end of this phase
        stable_idx[phase_num] = len(edges_list) - 1

    return edges_list, np.array(phase_list), np.array(t_list), stable_idx


def pairwise_distance_matrix(edges_list):
    n = len(edges_list)
    D = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(i + 1, n):
            d = hamming_edge_distance(edges_list[i], edges_list[j])
            D[i, j] = D[j, i] = d
    return D


def mds_embed(D):
    """Classical MDS to 2D. Returns (n, 2) coordinates."""
    n = D.shape[0]
    D2 = D.astype(float) ** 2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    # Top 2 eigenvalues
    idx = np.argsort(eigvals)[::-1][:2]
    L = np.diag(np.sqrt(np.maximum(eigvals[idx], 0)))
    coords = eigvecs[:, idx] @ L
    # Variance-explained for the title
    total_var = np.sum(np.maximum(eigvals, 0))
    var_explained = np.sum(np.maximum(eigvals[idx], 0)) / total_var if total_var > 0 else 0.0
    return coords, var_explained


def make_figure(result, baseline='initial', save_path=None):
    edges_list, phases, ts, stable_idx = collect_visits(result)
    D = pairwise_distance_matrix(edges_list)
    coords, var_explained = mds_embed(D)

    # Distance from baseline reference
    baseline_idx = stable_idx[0] if baseline == 'initial' else stable_idx[3]
    d_from_baseline = D[baseline_idx, :]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)

    # =========================================================================
    # Panel A: MDS embedding
    # =========================================================================
    axA = axes[0]
    for phase in (1, 2, 3):
        mask = phases == phase
        axA.plot(coords[mask, 0], coords[mask, 1], '-', color=PHASE_COLORS[phase],
                 alpha=0.4, lw=1.2)
        axA.scatter(coords[mask, 0], coords[mask, 1], s=22,
                    color=PHASE_COLORS[phase], alpha=0.85, edgecolors='none',
                    label=PHASE_LABELS[phase])
    # Stable configurations
    for k, marker, ms in [(0, 'X', 14), (1, '*', 18), (2, '*', 18), (3, '*', 18)]:
        idx = stable_idx[k]
        axA.scatter(coords[idx, 0], coords[idx, 1], marker=marker, s=ms ** 2,
                    facecolor='white', edgecolor='black', linewidth=1.5, zorder=10)
        label = f'$M_{k}$' if k > 0 else '$M_0$ (init)'
        axA.annotate(label, (coords[idx, 0], coords[idx, 1]),
                     xytext=(8, 8), textcoords='offset points', fontsize=11)
    axA.set_xlabel('MDS axis 1')
    axA.set_ylabel('MDS axis 2')
    axA.set_title(f'Trajectory in network space (MDS, {var_explained*100:.0f}% variance)')
    axA.grid(alpha=0.3)
    axA.legend(loc='best', fontsize=9)

    # =========================================================================
    # Panel B: distance-from-baseline time series
    # =========================================================================
    axB = axes[1]
    # Phase shading
    phase_starts = {p: ts[phases == p].min() for p in (1, 2, 3)}
    phase_ends = {p: ts[phases == p].max() for p in (1, 2, 3)}
    for p in (1, 2, 3):
        axB.axvspan(phase_starts[p], phase_ends[p], color=PHASE_COLORS[p], alpha=0.08)
    for p in (1, 2, 3):
        mask = phases == p
        axB.plot(ts[mask], d_from_baseline[mask], 'o-', color=PHASE_COLORS[p],
                 ms=4, lw=1.4, label=PHASE_LABELS[p], alpha=0.9)
    # Stable-config markers
    for k in (0, 1, 2, 3):
        idx = stable_idx[k]
        axB.scatter(ts[idx], d_from_baseline[idx], marker='*', s=180,
                    facecolor='white', edgecolor='black', zorder=10)
        label = f'$M_{k}$'
        axB.annotate(label, (ts[idx], d_from_baseline[idx]),
                     xytext=(6, 8), textcoords='offset points', fontsize=10)
    axB.set_xlabel('global swap index $t$')
    if baseline == 'initial':
        axB.set_ylabel('Hamming distance to $M_0$ (initial)')
        axB.set_title('Drift from initial configuration over time')
    else:
        axB.set_ylabel('Hamming distance to $M_3$ (final)')
        axB.set_title('Distance to final configuration over time')
    axB.grid(alpha=0.3)
    axB.legend(loc='best', fontsize=9)

    cfg = result['config']
    fig.suptitle(
        f"Shock experiment: n={cfg['n']}, c={cfg['c']}, cc={cfg['cc']}, "
        f"ms={cfg['max_swaps']}, mode={cfg['mode']}, "
        f"a={cfg['a_config']}, b={cfg['b_config']}, z={cfg['z_config']}, "
        f"aisi={cfg['aisi_spread']}, sw={cfg['sigma_w']}  |  "
        f"shutdown firm {result['firm_shutdown']} ({result['n_clients_at_shutdown']} clients)",
        fontsize=10
    )

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Wrote {save_path}")
    return fig


def print_summary(result):
    edges_list, phases, ts, stable_idx = collect_visits(result)
    D = pairwise_distance_matrix(edges_list)
    print("\n=== Shock-experiment summary ===")
    for k in (0, 1, 2, 3):
        idx = stable_idx[k]
        print(f"  M{k}:  visited at t={ts[idx]:>4d},  "
              f"d(M{k}, M0) = {D[stable_idx[0], idx]:>3d}")
    print(f"\n  d(M1, M2) = {D[stable_idx[1], stable_idx[2]]}")
    print(f"  d(M2, M3) = {D[stable_idx[2], stable_idx[3]]}")
    print(f"  d(M1, M3) = {D[stable_idx[1], stable_idx[3]]}")
    for p in (1, 2, 3):
        n_visits = (phases == p).sum()
        print(f"  phase {p}: {n_visits} configurations visited "
              f"({result[f'phase{p}']['rounds']} rounds, "
              f"{result[f'phase{p}']['total_rewirings']} rewirings)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='pickle path written by shock_experiment.py')
    p.add_argument('--baseline', choices=['initial', 'final'], default='initial')
    p.add_argument('--output', default=None,
                   help='figure path; defaults to <input>.png')
    args = p.parse_args()

    with open(args.input, 'rb') as f:
        result = pickle.load(f)

    print_summary(result)

    save_path = args.output or os.path.splitext(args.input)[0] + '.png'
    make_figure(result, baseline=args.baseline, save_path=save_path)


if __name__ == '__main__':
    main()
