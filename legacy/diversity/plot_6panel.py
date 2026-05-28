"""6-panel manuscript figure: diversity (same_tech_dif_init) vs three
heterogeneity sources at two action regimes.

Layout (rows = ms, cols = swept axis):
                  aisi sweep      sigma_w sweep    z_width sweep
    ms=1     (a)                  (b)               (c)
    ms=cc=4  (d)                  (e)               (f)

Series per panel: 4 b-regimes + 1 extra
    1.  CRS hom            a=hom 0.5,  b=hom 1.0      (z varies on right panel)
    2.  DRS hom            a=hom 0.5,  b=hom 0.9
    3.  IRS hom            a=hom 0.5,  b=hom 1.1
    4.  HRS                a=hom 0.5,  b=unif 0.9:1.1
    5.  HRS + 2 hetero     same a/b as HRS, plus 2 fixed heterogeneities
                            (per panel: see launcher comments)

Reads CSVs from one or more directories (e.g. results_6panel_v1/ +
results_4panel_v1/ for reuse).

X-axis values are read directly from the data:
    aisi_spread (left), sigma_w (center), z_width (right; computed from z_config).

Usage:
    python scripts/plot_6panel.py results_6panel_v1
    python scripts/plot_6panel.py results_6panel_v1 results_4panel_v1
    python scripts/plot_6panel.py results_6panel_v1 --output paper/fig_6panel.png
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def cfg_str(s):
    try:
        d = json.loads(s)
        if d['mode'] == 'homogeneous':
            return f"hom:{d['value']}"
        return f"unif:{d['min']}:{d['max']}"
    except Exception:
        return 'NA'


def z_width(z_short):
    """Half-width of the uniform z distribution centered on 1.0; 0 for homogeneous."""
    if pd.isna(z_short):
        return np.nan
    if z_short.startswith('hom:'):
        return 0.0
    if z_short.startswith('unif:'):
        _, lo, hi = z_short.split(':')
        return round((float(hi) - float(lo)) / 2.0, 4)
    return np.nan


# -----------------------------------------------------------------------------
# Series + panel definitions
# -----------------------------------------------------------------------------

# (sid, label, a_short, b_short, color, marker)
BASE_SERIES = [
    ('CRS', r'CRS hom ($b=1.0$)',          'hom:0.5', 'hom:1.0',      'C0', 'o'),
    ('DRS', r'DRS hom ($b=0.9$)',          'hom:0.5', 'hom:0.9',      'C2', 's'),
    ('IRS', r'IRS hom ($b=1.1$)',          'hom:0.5', 'hom:1.1',      'C3', '^'),
    ('HRS', r'HRS ($b\sim U[0.9,1.1]$)',   'hom:0.5', 'unif:0.9:1.1', 'C1', 'D'),
]

EXTRA = dict(label='HRS + 2 het. fixed', a='hom:0.5', b='unif:0.9:1.1',
             color='C5', marker='X')

PANELS = [
    {
        'col_idx': 0,
        'x_col': 'aisi_spread',
        'x_label': 'AiSi spread',
        'base_fixed':  {'sigma_w': 0.0,  'z_width': 0.0},
        'extra_fixed': {'sigma_w': 0.1,  'z_width': 0.1},
    },
    {
        'col_idx': 1,
        'x_col': 'sigma_w',
        'x_label': r'Link noise $\sigma_w$',
        'base_fixed':  {'aisi_spread': 0.0,  'z_width': 0.0},
        'extra_fixed': {'aisi_spread': 0.05, 'z_width': 0.1},
    },
    {
        'col_idx': 2,
        'x_col': 'z_width',
        'x_label': r'$z$ half-width',
        'base_fixed':  {'aisi_spread': 0.0,  'sigma_w': 0.0},
        'extra_fixed': {'aisi_spread': 0.05, 'sigma_w': 0.1},
    },
]


# -----------------------------------------------------------------------------
# Data loading & filtering
# -----------------------------------------------------------------------------

def load_data(dirs):
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  warning: {d} is not a directory, skipping")
            continue
        paths.extend(sorted(glob.glob(os.path.join(d, '*.csv'))))
    if not paths:
        raise FileNotFoundError(f"No CSVs in {dirs}")
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if 'a_config' not in d.columns or 'series' not in d.columns:
            continue
        dfs.append(d)
    if not dfs:
        raise RuntimeError("No diversity-format CSVs found")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['a_short'] = df['a_config'].apply(cfg_str)
    df['b_short'] = df['b_config'].apply(cfg_str)
    df['z_short'] = df['z_config'].apply(cfg_str)
    df['z_width'] = df['z_short'].apply(z_width)
    df = df[df['series'] == 'same_tech_dif_init'].copy()
    df = df[(df['n'] == 100) & (df['cc'] == 4)].copy()
    print(f"Loaded {len(paths)} CSVs -> {len(df)} dif_init rows at n=100, cc=4")
    return df


def filter_series_curve(df, a_short, b_short, ms, x_col, fixed):
    """Pick rows matching (a, b, ms) and the panel's fixed-axis constraints,
    then aggregate diversity by x_col across tech matrices.
    Returns DataFrame with columns [x_col, mean, sem, count] or None.
    """
    sub = df[(df['a_short'] == a_short) &
             (df['b_short'] == b_short) &
             (df['max_swaps'] == ms)]
    for col, val in fixed.items():
        sub = sub[np.isclose(sub[col], val, atol=1e-6)]
    if len(sub) == 0:
        return None
    g = (sub.groupby(x_col)['diversity']
              .agg(['mean', 'sem', 'count'])
              .sort_index()
              .reset_index())
    return g


# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------

def plot_panel(ax, df, panel, ms):
    """Render one panel: 4 base + 1 extra series."""
    for sid, label, a, b, color, marker in BASE_SERIES:
        g = filter_series_curve(df, a, b, ms, panel['x_col'], panel['base_fixed'])
        if g is None or len(g) == 0:
            continue
        ax.errorbar(g[panel['x_col']], g['mean'],
                    yerr=1.96 * g['sem'].fillna(0),
                    fmt=marker + '-', color=color, lw=1.5, ms=5,
                    capsize=2.5, label=label, alpha=0.95)

    g = filter_series_curve(df, EXTRA['a'], EXTRA['b'], ms,
                             panel['x_col'], panel['extra_fixed'])
    if g is not None and len(g) > 0:
        ax.errorbar(g[panel['x_col']], g['mean'],
                    yerr=1.96 * g['sem'].fillna(0),
                    fmt=EXTRA['marker'] + '--', color=EXTRA['color'],
                    lw=1.5, ms=5, capsize=2.5,
                    label=EXTRA['label'], alpha=0.95)


def coverage_report(df):
    print("\n=== Coverage per (panel, series, ms) ===")
    for ms in [1, 4]:
        for panel in PANELS:
            for sid, label, a, b, *_ in BASE_SERIES:
                g = filter_series_curve(df, a, b, ms, panel['x_col'],
                                         panel['base_fixed'])
                if g is None:
                    continue
                xs = list(g[panel['x_col']].values)
                tot = int(g['count'].sum())
                print(f"  ms={ms}  {panel['x_col']:13} {sid:3} | "
                      f"{len(xs)} pts, {tot} tech-rows  "
                      f"x={xs}")
            # Extra
            g = filter_series_curve(df, EXTRA['a'], EXTRA['b'], ms,
                                     panel['x_col'], panel['extra_fixed'])
            if g is not None and len(g) > 0:
                xs = list(g[panel['x_col']].values)
                tot = int(g['count'].sum())
                print(f"  ms={ms}  {panel['x_col']:13} EXT | "
                      f"{len(xs)} pts, {tot} tech-rows  "
                      f"x={xs}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dirs', nargs='+',
                   help='One or more directories containing diversity CSVs.')
    p.add_argument('--output', default=None,
                   help='Output PNG path (default: <first data_dir>/figure_6panel.png)')
    args = p.parse_args()

    df = load_data(args.data_dirs)
    coverage_report(df)

    plt.rcParams.update({
        'font.size':       10,
        'axes.titlesize':  11,
        'axes.labelsize':  10,
        'xtick.labelsize':  9,
        'ytick.labelsize':  9,
        'legend.fontsize':  8.5,
    })

    fig, axes = plt.subplots(2, 3, figsize=(13, 7),
                              constrained_layout=True, sharey=True)

    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for row, ms in enumerate([1, 4]):
        ms_label = r'$ms=1$' if ms == 1 else r'$ms=cc=4$'
        for panel in PANELS:
            ax = axes[row, panel['col_idx']]
            plot_panel(ax, df, panel, ms)
            letter = letters[row * 3 + panel['col_idx']]
            ax.set_title(f"{letter} {ms_label},   x = {panel['x_label']}",
                         loc='left')
            ax.set_xlabel(panel['x_label'])
            ax.set_ylim(-0.03, 1.03)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
            ax.grid(alpha=0.3)

    # Y-axis label only on left column; right two columns hide labels (sharey).
    axes[0, 0].set_ylabel('Equilibrium diversity')
    axes[1, 0].set_ylabel('Equilibrium diversity')
    for r in range(2):
        for c in [1, 2]:
            axes[r, c].set_ylabel('')
            axes[r, c].tick_params(labelleft=False)

    # One legend, placed in the (a) panel's middle right.
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        axes[0, 0].legend(h, l, loc='center right', framealpha=0.95)

    out = args.output or os.path.join(args.data_dirs[0], 'figure_6panel.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
