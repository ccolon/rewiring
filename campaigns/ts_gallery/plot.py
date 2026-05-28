"""Single-trajectory time-series figure for the rewiring simulation.

Runs ONE `run_unified_simulation` from the CLI args, then writes TWO
files to THIS SCRIPT'S DIRECTORY:
  <stem>.plot.pkl  -- small render-ready cache (deduped t/distance arrays,
                      MDS coords, cycle info, rewire events). Cosmetic
                      replots load this and skip the O(N^2) edge-distance
                      matrix + O(N^3) MDS eigendecomposition.
  <stem>.png       -- the 3-panel figure.

The full raw simulation payload is no longer persisted; the trace is
distilled in memory into `<stem>.plot.pkl` and discarded. To replot from a
prior run, point `--from_pkl` at the saved `<stem>.plot.pkl`.

Legacy compatibility: `--from_pkl <stem>.pkl` (a raw payload from an
earlier version of this script) is still accepted -- the cache is built
from it once and written as a sidecar next to the raw file. The raw `.pkl`
itself is never produced by this script anymore.

Layout (no suptitle):
    Left column (split vertically):
        (a) top    -- distance d(M_t, M_0) over global step t
        (b) bottom -- rewire events (firm id vs t) as black squares
    Right column:
        (c) MDS embedding of visited configurations (classical MDS on the
            cosine edge-distance matrix), colored by visit order.

CLI mirrors `scripts/shock_experiment.py`. Output filenames encode the key
params, so a single directory becomes a self-documenting gallery.

Usage (from anywhere):
    python results/ts_gallery/plot_ts.py
    python results/ts_gallery/plot_ts.py --mode aa --b_config homogeneous:0.95
    python results/ts_gallery/plot_ts.py --n 30 --seed 7 --max_swaps 2

`limited` / `naive_limited` mode use a per-firm tier (tau_i). The CLI
follows `scripts/visibility_study.py`'s canonical (Poisson) convention:
    --tier_mean   mean tier visibility tau
    --tau_mode    homo   -> every firm gets round(tier_mean)
                  hetero -> tier_i ~ Poisson(lambda=tier_mean), per firm

Examples:
    # Homogeneous tau = 2
    python results/ts_gallery/plot_ts.py --mode limited --tier_mean 2
    # Heterogeneous Poisson(lambda=2)
    python results/ts_gallery/plot_ts.py --mode limited \\
        --tier_mean 2 --tau_mode hetero
"""
import argparse
import math
import os
import pickle
import re
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
# Tracked code in campaigns/ts_gallery/; pickle caches + figures land in the
# mirroring results dir.
OUTPUT_DIR = os.path.join(REPO_ROOT, 'results', 'ts_gallery')
os.makedirs(OUTPUT_DIR, exist_ok=True)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from rewiring.networks import generate_base_network, generate_random_initial_network
from rewiring.simulation import run_unified_simulation


# Matches plot_shock.py's typographic scaling.
FONT_SCALE = 1.8
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


# -----------------------------------------------------------------------------
# Param parsing (mirrors shock_experiment.py)
# -----------------------------------------------------------------------------
def _parse_param(s, n, rng):
    """Parse 'homogeneous:V' or 'uniform:LO:HI' into a length-n array."""
    parts = s.split(':')
    mode = parts[0]
    if mode == 'homogeneous':
        return np.full(n, float(parts[1]))
    if mode == 'uniform':
        lo, hi = float(parts[1]), float(parts[2])
        return rng.uniform(lo, hi, n)
    raise ValueError(f"Unknown param mode: {mode!r}")


def _build_tier_array(n, tier_mean, tau_mode, rng):
    """Per-firm tier (tau_i) array. Mirrors `visibility_study.py`'s
    canonical (Poisson) branch.

    tau_mode='homo'   -> full(n, round(tier_mean))
    tau_mode='hetero' -> tier_i ~ Poisson(lambda=tier_mean), n iid draws.
                         (Std is implicitly sqrt(lambda); no separate knob.)
    """
    if tau_mode == 'homo':
        return np.full(n, int(round(tier_mean)), dtype=int)
    if tau_mode == 'hetero':
        if tier_mean <= 0:
            return np.zeros(n, dtype=int)
        return rng.poisson(lam=tier_mean, size=n).astype(int)
    raise ValueError(f"Unknown tau_mode={tau_mode!r}")


# -----------------------------------------------------------------------------
# Distance / MDS helpers (lifted from plot_shock.py so this script is standalone)
# -----------------------------------------------------------------------------
def edges_to_set(arr):
    """Convert a (K, 2) edge array into a frozenset of (supplier, buyer) tuples."""
    return frozenset((int(s), int(b)) for s, b in arr)


def cosine_edge_distance(e1, e2):
    """1 - Ochiai (cosine) distance on binary edge sets, in [0, 1]."""
    if not e1 or not e2:
        return 1.0
    return 1.0 - len(e1 & e2) / math.sqrt(len(e1) * len(e2))


def collect_visits(trace):
    """Walk swap_edges/price_steps and drop consecutive duplicate snapshots.

    Returns (edges_list, t_arr): the first entry is M_0 at t=0.
    """
    edges_list = []
    t_list = []
    swap_edges = trace['swap_edges']
    steps = trace['price_steps']
    for k, (e, t_local) in enumerate(zip(swap_edges, steps)):
        if k > 0 and edges_to_set(e) == edges_to_set(swap_edges[k - 1]):
            continue
        edges_list.append(edges_to_set(e))
        t_list.append(int(t_local))
    return edges_list, np.array(t_list)


def pairwise_distance_matrix(edges_list):
    n = len(edges_list)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine_edge_distance(edges_list[i], edges_list[j])
            D[i, j] = D[j, i] = d
    return D


def detect_round_cycle_period(round_edges_list, max_period=10):
    """Smallest k in [2, max_period] such that the last 2k round-end
    configurations form a period-k cycle, else None.

    Mirrors the simulation's `_detect_period` at the round level, so the
    detection works even when the run was made with `detect_cycles=False`.
    """
    if not round_edges_list:
        return None
    sigs = [edges_to_set(e) for e in round_edges_list]
    n_sigs = len(sigs)
    for k in range(2, min(max_period, n_sigs // 2) + 1):
        if sigs[-2 * k:-k] == sigs[-k:]:
            return k
    return None


def _idx_of_config(edge_set, edges_list, prefer='last'):
    """Index of `edge_set` in `edges_list` (None if absent).
    `prefer` selects the first or last matching occurrence."""
    if prefer == 'last':
        for i in range(len(edges_list) - 1, -1, -1):
            if edges_list[i] == edge_set:
                return i
    else:
        for i, e in enumerate(edges_list):
            if e == edge_set:
                return i
    return None


def build_rolling_event_count(events, t_max, window, n_bins=400):
    """Aggregate (across-firm) rolling count of rewire events.

    Returns (t_centers, counts):
        t_centers: 1D bin-center array along t.
        counts:    1D float array; counts[k] = number of rewire events
                   (summed over all firms) inside a `window`-wide slice of
                   time centered on bin k. Box convolution, `mode='same'`
                   (zero-padded edges).
    """
    if not events or t_max <= 0:
        return (np.array([0.0, max(t_max, 1.0)]),
                np.array([0.0, 0.0]))
    n_bins = max(40, min(n_bins, int(t_max) + 1))
    bin_edges = np.linspace(0, t_max, n_bins + 1)
    t_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    ts_evt = np.array([e['t'] for e in events], dtype=float)
    ts_evt = ts_evt[(ts_evt >= 0) & (ts_evt <= t_max)]
    counts, _ = np.histogram(ts_evt, bins=bin_edges)
    bin_width = t_max / n_bins
    w_bins = max(1, int(round(window / bin_width)))
    w_bins = min(w_bins, n_bins)
    counts = counts.astype(float)
    if w_bins > 1:
        counts = np.convolve(counts, np.ones(w_bins), mode='same')
    return t_centers, counts


def mds_embed(D):
    """Classical MDS to 2D. Returns (n, 2) coordinates and variance explained."""
    n = D.shape[0]
    if n < 2:
        return np.zeros((n, 2)), 0.0
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


# -----------------------------------------------------------------------------
# Simulation runner
# -----------------------------------------------------------------------------
def run(args):
    rng = np.random.default_rng(args.seed)
    a = _parse_param(args.a_config, args.n, rng)
    b = _parse_param(args.b_config, args.n, rng)
    z = _parse_param(args.z_config, args.n, rng)

    tier_arr = None
    if args.mode in ('limited', 'naive_limited'):
        # Independent RNG so tier draws don't shift the a/b/z draws above.
        tier_rng = np.random.default_rng(args.seed + 500)
        tier_arr = _build_tier_array(args.n, args.tier_mean, args.tau_mode,
                                     tier_rng)

    base_ns = generate_base_network(
        n=args.n, c=args.c, cc=args.cc, aisi_spread=args.aisi,
        seed=args.seed, a=a, b=b, sigma_w=args.sigma_w,
    )
    init_ns = generate_random_initial_network(
        n=args.n, Wbar=base_ns['Wbar'], AiSi=base_ns['AiSi'],
        seed=args.seed + 1000,
    )
    result = run_unified_simulation(
        init_ns, a, b, z,
        mode=args.mode, seed=args.seed + 2000,
        max_swaps=args.max_swaps, nb_rounds=args.nb_rounds,
        tier=tier_arr,
        trace=True, synchronous=args.synchronous,
        detect_cycles=not args.no_detect_cycles,
    )
    return {
        'config': vars(args),
        'a': a, 'b': b, 'z': z,
        'tier_arr': tier_arr,
        'base_ns': {k: base_ns[k] for k in ('M0', 'Wbar', 'supplier_id_list',
                                             'alternate_supplier_id_list',
                                             'nb_suppliers')},
        'init_supplier_list': [list(s) for s in init_ns['supplier_id_list']],
        'result': result,
    }


# -----------------------------------------------------------------------------
# Plot data: small, render-ready cache (separate from the heavy raw payload)
# -----------------------------------------------------------------------------
# Bump when build_plot_data's schema changes so stale caches get recomputed.
PLOT_DATA_VERSION = 1


def build_plot_data(payload, ma_window=None):
    """Distil everything `render_figure` needs from a full simulation payload.

    Does the heavy work once (consecutive-dedup, pairwise edge-distance
    matrix, classical MDS, post-hoc cycle detection) and returns a small
    dict that survives a pickle round-trip in milliseconds. Cosmetic
    replots reuse the cache and skip all of this.

    `ma_window` is just stored on the side; the heatmap itself is
    rebuilt cheaply from `events` inside `render_figure`, so changing the
    window at replot time stays in the fast path.
    """
    result = payload['result']
    trace = result['trace']
    n = payload['config']['n']
    if ma_window is None:
        ma_window = int(n)

    edges_list, ts = collect_visits(trace)
    D = pairwise_distance_matrix(edges_list)
    coords, var_explained = mds_embed(D)
    d_from_M0 = D[0, :]

    is_fixed_point = bool(result.get('converged'))
    cycle_period = None
    cycle_positions = []
    if not is_fixed_point:
        round_edges = trace.get('edges', [])
        cycle_period = detect_round_cycle_period(round_edges, max_period=10)
        if cycle_period is not None:
            for round_state in round_edges[-cycle_period:]:
                idx = _idx_of_config(edges_to_set(round_state),
                                     edges_list, prefer='last')
                if idx is not None:
                    cycle_positions.append(idx)

    events = [{'t': int(e['t']), 'firm': int(e['firm'])}
              for e in trace.get('rewire_events', [])]
    return {
        '_version':        PLOT_DATA_VERSION,
        'n':               int(n),
        'ts':              ts,
        'd_from_M0':       d_from_M0,
        'coords':          coords,
        'var_explained':   float(var_explained),
        'events':          events,
        'is_fixed_point':  is_fixed_point,
        'cycle_period':    cycle_period,
        'cycle_positions': cycle_positions,
        'ma_window':       int(ma_window),
    }


# -----------------------------------------------------------------------------
# Figure (pure matplotlib; reads only from a `build_plot_data` dict)
# -----------------------------------------------------------------------------
def render_figure(plot_data, save_path):
    n               = plot_data['n']
    ts              = plot_data['ts']
    d_from_M0       = plot_data['d_from_M0']
    coords          = plot_data['coords']
    var_explained   = plot_data['var_explained']
    events          = plot_data['events']
    is_fixed_point  = plot_data['is_fixed_point']
    cycle_period    = plot_data['cycle_period']
    cycle_positions = plot_data['cycle_positions']

    with plt.rc_context(PLOT_RCPARAMS):
        fig = plt.figure(figsize=(13, 5.8), constrained_layout=True)
        outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.0, 1.0])
        left = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer[0],
            height_ratios=[1.4, 1.0], hspace=0.05,
        )
        ax_ts  = fig.add_subplot(left[0])
        ax_re  = fig.add_subplot(left[1], sharex=ax_ts)
        ax_mds = fig.add_subplot(outer[1])

        # ---------- (a) Distance TS ----------
        ax_ts.plot(ts, d_from_M0, 'o-', color='#1f77b4', ms=4, lw=1.4,
                   alpha=0.92)
        ax_ts.set_ylabel(r'Distance to $M_0$')
        ax_ts.set_title('(a) Drift over time', loc='left')
        ax_ts.grid(alpha=0.3)
        ax_ts.tick_params(labelbottom=False)

        # ---------- (b) Rewire events + aggregated rolling count ----------
        # Background: total rewires across all firms in a `ma_window`-wide
        # slice of t, drawn as a filled curve on a twin right-side axis.
        # Foreground: the raw black-square scatter on the main axis.
        ma_window = int(plot_data.get('ma_window', n))
        t_max = int(ts.max()) if ts.size else 1
        t_centers, rolling = build_rolling_event_count(
            events, t_max=max(t_max, 1), window=ma_window,
        )

        ax_re_bg = ax_re.twinx()
        # Put the twin axis BEHIND the main axis so the scatter stays on
        # top of the filled curve (standard matplotlib trick).
        ax_re.set_zorder(ax_re_bg.get_zorder() + 1)
        ax_re.patch.set_visible(False)

        FILL_COLOR = '#ff8c69'  # salmon
        LINE_COLOR = '#c14b2a'  # deeper salmon
        if rolling.size and rolling.max() > 0:
            ax_re_bg.fill_between(t_centers, 0, rolling,
                                  color=FILL_COLOR, alpha=0.30,
                                  linewidth=0, zorder=1)
            ax_re_bg.plot(t_centers, rolling, color=LINE_COLOR,
                          lw=1.2, alpha=0.85, zorder=1)
            ax_re_bg.set_ylim(bottom=0, top=rolling.max() * 1.05)
        else:
            ax_re_bg.set_ylim(bottom=0, top=1)
        # Hide all twin-axis chrome: the curve is decorative, no scale
        # or legend entry.
        ax_re_bg.tick_params(axis='y', length=0, labelleft=False,
                             labelright=False)
        for spine in ax_re_bg.spines.values():
            spine.set_visible(False)

        if events:
            xs = [e['t'] for e in events]
            ys = [e['firm'] for e in events]
            ax_re.scatter(xs, ys, c='black', marker='s', s=8,
                          edgecolors='black', linewidths=0.4, zorder=3)
        ax_re.set_xlabel(r'Time step $t$')
        ax_re.set_ylabel('Rewiring firm id')
        ax_re.set_ylim(-0.5, n - 0.5)
        if n <= 15:
            ax_re.set_yticks(range(n))
        ax_re.set_title('(b) Rewire events', loc='left')
        ax_re.grid(alpha=0.3)
        ax_ts.set_xlim(0, max(t_max, 1))

        # ---------- (c) MDS embedding ----------
        ax_mds.plot(coords[:, 0], coords[:, 1], '-', color='gray',
                    alpha=0.4, lw=1.0)
        sc = ax_mds.scatter(coords[:, 0], coords[:, 1], c=ts,
                            cmap='viridis', s=36, edgecolors='none')
        ax_mds.scatter(coords[0, 0], coords[0, 1], marker='X', s=240,
                       facecolor='white', edgecolor='black',
                       linewidth=1.6, zorder=10)
        ax_mds.annotate(r'$M_0$', (coords[0, 0], coords[0, 1]),
                        xytext=(8, 8), textcoords='offset points')

        if is_fixed_point:
            ax_mds.scatter(coords[-1, 0], coords[-1, 1], marker='*', s=320,
                           facecolor='white', edgecolor='black',
                           linewidth=1.6, zorder=10)
            ax_mds.annotate(r'$M_\mathrm{end}$',
                            (coords[-1, 0], coords[-1, 1]),
                            xytext=(8, 8), textcoords='offset points')
        elif cycle_positions:
            ANNOTATE_CAP = 6
            for j, pos in enumerate(cycle_positions):
                ax_mds.scatter(coords[pos, 0], coords[pos, 1], marker='*',
                               s=320, facecolor='white', edgecolor='black',
                               linewidth=1.6, zorder=10)
            if cycle_period <= ANNOTATE_CAP:
                for j, pos in enumerate(cycle_positions):
                    dx, dy = (8, 8) if j % 2 == 0 else (-32, -14)
                    ax_mds.annotate(
                        rf'$M_\mathrm{{cycle}}^{{({j + 1})}}$',
                        (coords[pos, 0], coords[pos, 1]),
                        xytext=(dx, dy), textcoords='offset points')
            else:
                pos = cycle_positions[-1]
                ax_mds.annotate(
                    rf'$M_\mathrm{{cycle}}\;(k={cycle_period})$',
                    (coords[pos, 0], coords[pos, 1]),
                    xytext=(8, 8), textcoords='offset points')

        ax_mds.set_xlabel('MDS axis 1')
        ax_mds.set_ylabel('MDS axis 2')
        ax_mds.set_title(f'(c) Network space '
                         f'(MDS, {var_explained*100:.0f}% variance)',
                         loc='left')
        ax_mds.grid(alpha=0.3)
        cbar = fig.colorbar(sc, ax=ax_mds, pad=0.02, shrink=0.85)
        cbar.set_label(r'$t$')

        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Wrote {save_path}")


# -----------------------------------------------------------------------------
# Output naming
# -----------------------------------------------------------------------------
def _fmt_num(s):
    """Strip trailing zeros: '2.0' -> '2', '0.50' -> '0.5'. Pass-through on
    non-numeric strings."""
    try:
        return f"{float(s):g}"
    except ValueError:
        return s


def _compact_config(s):
    """homogeneous:V -> '<V>'   (homo is implicit; trailing zeros trimmed)
       uniform:LO:HI -> 'u<LO>-<HI>'"""
    parts = s.split(':')
    mode = parts[0]
    if mode == 'homogeneous':
        return _fmt_num(parts[1])
    if mode == 'uniform':
        return f"u{_fmt_num(parts[1])}-{_fmt_num(parts[2])}"
    return s


def default_stem(args):
    parts = [
        f"ts_n{args.n}",
        f"c{args.c}",
        f"cc{args.cc}",
        f"ms{args.max_swaps}",
        args.mode,
        f"a-{_compact_config(args.a_config)}",
        f"b-{_compact_config(args.b_config)}",
        f"z-{_compact_config(args.z_config)}",
    ]
    if args.mode in ('limited', 'naive_limited'):
        tau_suffix = 'H' if args.tau_mode == 'homo' else 'P'
        parts.append(f"tau{args.tier_mean:g}{tau_suffix}")
    if args.aisi:
        parts.append(f"aisi{args.aisi:g}")
    if args.sigma_w:
        parts.append(f"sw{args.sigma_w:g}")
    if args.synchronous:
        parts.append("sync")
    parts.append(f"nr{args.nb_rounds}")
    parts.append(f"seed{args.seed}")
    return "_".join(parts)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description='Single-trajectory TS+rewire+MDS figure for one rewiring run.'
    )
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--n', type=int, default=50)
    p.add_argument('--c', type=int, default=4)
    p.add_argument('--cc', type=int, default=4)
    p.add_argument('--max_swaps', type=int, default=1)
    p.add_argument('--mode', choices=['full', 'aa', 'limited', 'naive_limited'],
                   default='full')
    p.add_argument('--a_config', default='homogeneous:0.5')
    p.add_argument('--b_config', default='homogeneous:0.9')
    p.add_argument('--z_config', default='homogeneous:1.0')
    p.add_argument('--aisi', type=float, default=0.0)
    p.add_argument('--sigma_w', type=float, default=0.0)
    p.add_argument('--nb_rounds', type=int, default=200)
    # --- tier (tau_i) controls; only consumed when mode in (limited, naive_limited)
    p.add_argument('--tier_mean', type=float, default=2.0,
                   help='Mean tier visibility tau (ignored unless mode in '
                        "{'limited','naive_limited'}). For tau_mode=hetero "
                        'this is the Poisson lambda; for tau_mode=homo it is '
                        'rounded to the nearest int.')
    p.add_argument('--tau_mode', choices=['homo', 'hetero'], default='homo',
                   help='homo:   tier_arr = full(n, round(tier_mean)).  '
                        'hetero: tier_i ~ Poisson(lambda=tier_mean) per firm '
                        '(matches the canonical visibility-study branch).')
    p.add_argument('--synchronous', action='store_true',
                   help='Jacobi update instead of the default Gauss-Seidel.')
    p.add_argument('--no_detect_cycles', action='store_true',
                   help='Disable cycle detection (keeps the trace logging '
                        'past the first detected period-k cycle).')
    p.add_argument('--output', default=None,
                   help='Output stem (no extension). Default: encodes key '
                        "params; resulting files land in this script's dir.")
    p.add_argument('--ma_window', type=int, default=None,
                   help='Moving-average window (in t-steps) for the rewire-'
                        'rate background in panel (b). Default: n (matches '
                        'one Gauss-Seidel round). On --from_pkl, overrides '
                        'the value stored in the .plot.pkl cache.')
    p.add_argument('--from_pkl', default=None,
                   help='Skip the simulation and replot from an existing '
                        'pickle. Preferred: <stem>.plot.pkl (render-ready '
                        'cache). Also accepts a legacy raw <stem>.pkl from '
                        'an older version of this script -- the cache is '
                        'built once and written as a sidecar next to it.')
    p.add_argument('--rebuild_cache', action='store_true',
                   help='With --from_pkl pointing at a legacy raw .pkl: '
                        'ignore an existing .plot.pkl sidecar and rebuild '
                        'it.')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    if args.from_pkl is not None:
        plot_data, out_png = _load_for_replot(args)
    else:
        plot_data, out_png = _run_and_cache(args)

    render_figure(plot_data, out_png)


# -----------------------------------------------------------------------------
# Main-flow helpers
# -----------------------------------------------------------------------------
def _plot_pkl_path_for(raw_pkl_path):
    """Sidecar path for a raw .pkl: '<dir>/<stem>.plot.pkl'."""
    d = os.path.dirname(os.path.abspath(raw_pkl_path))
    base = os.path.basename(raw_pkl_path)
    stem = base[:-len('.pkl')] if base.endswith('.pkl') else \
        os.path.splitext(base)[0]
    return os.path.join(d, stem + '.plot.pkl')


def _run_and_cache(args):
    """Fresh run path: simulate, distil to plot_data, write only the
    sidecar cache + PNG. The raw payload is discarded after distillation."""
    stem = args.output if args.output else default_stem(args)
    out_plot_pkl = os.path.join(OUTPUT_DIR, stem + '.plot.pkl')
    out_png      = os.path.join(OUTPUT_DIR, stem + '.png')

    tier_msg = ""
    if args.mode in ('limited', 'naive_limited'):
        if args.tau_mode == 'hetero':
            tier_msg = (f"  tau_mode=hetero(Poisson)"
                        f"  tier_mean={args.tier_mean}")
        else:
            tier_msg = (f"  tau_mode=homo"
                        f"  tier={int(round(args.tier_mean))}")
    print(f"Running mode={args.mode}  n={args.n}  b_config={args.b_config}  "
          f"seed={args.seed}  max_swaps={args.max_swaps}{tier_msg} ...")
    payload = run(args)
    if payload['tier_arr'] is not None:
        ta = payload['tier_arr']
        print(f"  tier_arr: mean={ta.mean():.2f}  std={ta.std():.2f}  "
              f"min={ta.min()}  max={ta.max()}")
    r = payload['result']
    print(f"  rounds={r['rounds']}  converged={r['converged']}  "
          f"cycle_period={r['cycle_period']}  "
          f"total_rewirings={r['total_rewirings']}  "
          f"events_traced={len(r['trace']['rewire_events'])}")

    ma_window = args.ma_window if args.ma_window is not None else args.n
    plot_data = build_plot_data(payload, ma_window=ma_window)
    # Drop the heavy payload now that the plot cache has everything we
    # need; the raw trace is intentionally not persisted to disk.
    del payload
    with open(out_plot_pkl, 'wb') as f:
        pickle.dump(plot_data, f)
    print(f"Wrote {out_plot_pkl}  (render-ready cache)")

    return plot_data, out_png


def _load_for_replot(args):
    """--from_pkl path: load the render-ready cache if available, else
    rebuild from the raw payload (and refresh the cache for next time).
    Returns (plot_data, out_png)."""
    src = args.from_pkl
    stem_in = os.path.basename(src)
    for suffix in ('.plot.pkl', '.pkl'):
        if stem_in.endswith(suffix):
            stem_in = stem_in[:-len(suffix)]
            break
    stem_out = args.output if args.output else stem_in
    out_png = os.path.join(OUTPUT_DIR, stem_out + '.png')

    # If the user pointed directly at a .plot.pkl, load it and done.
    if src.endswith('.plot.pkl'):
        with open(src, 'rb') as f:
            plot_data = pickle.load(f)
        print(f"Loaded plot cache: {src}")
        if plot_data.get('_version') != PLOT_DATA_VERSION:
            print(f"  warning: plot cache version "
                  f"{plot_data.get('_version')} != {PLOT_DATA_VERSION}; "
                  f"render may be inconsistent.")
        if args.ma_window is not None:
            plot_data['ma_window'] = int(args.ma_window)
        return plot_data, out_png

    # Otherwise --from_pkl is a raw .pkl. Prefer its sidecar cache.
    plot_path = _plot_pkl_path_for(src)
    if (not args.rebuild_cache) and os.path.exists(plot_path):
        try:
            with open(plot_path, 'rb') as f:
                cached = pickle.load(f)
            if cached.get('_version') == PLOT_DATA_VERSION:
                print(f"Loaded plot cache: {plot_path}")
                if args.ma_window is not None:
                    cached['ma_window'] = int(args.ma_window)
                return cached, out_png
            print(f"Plot cache version mismatch "
                  f"({cached.get('_version')} != {PLOT_DATA_VERSION}); "
                  f"rebuilding.")
        except Exception as ex:
            print(f"Plot cache unreadable ({ex}); rebuilding.")

    with open(src, 'rb') as f:
        payload = pickle.load(f)
    print(f"Loaded {src}")
    # Use --ma_window if provided, else the legacy payload's n.
    legacy_n = payload.get('config', {}).get('n')
    ma_window = (args.ma_window if args.ma_window is not None
                 else (legacy_n if legacy_n is not None else None))
    plot_data = build_plot_data(payload, ma_window=ma_window)
    try:
        with open(plot_path, 'wb') as f:
            pickle.dump(plot_data, f)
        print(f"Wrote {plot_path}  (render-ready cache)")
    except OSError as ex:
        print(f"Could not write plot cache to {plot_path}: {ex}")
    return plot_data, out_png


if __name__ == '__main__':
    main()
