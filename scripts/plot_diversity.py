"""6-panel diversity figure (n=100, cc=4, ms=1).

Layout (rows = series families, cols = swept axis):

                     aisi sweep          sigma_w sweep        z_width sweep
    Top: HOMO       (a) CRS,DRS,IRS    (b) CRS,DRS,IRS      (c) CRS,DRS,IRS
    Bot: HETERO     (d) HRS variants   (e) HRS variants     (f) HRS variants

Top-row series (3 base regimes):
    CRS hom:   a=hom 0.5, b=hom 1.0, z=hom 1.0
    DRS hom:   a=hom 0.5, b=hom 0.9, z=hom 1.0
    IRS hom:   a=hom 0.5, b=hom 1.1, z=hom 1.0

Bottom-row series (HRS + 0/1/2 additional fixed heterogeneity sources):

    Each series is shown on the panel(s) where its swept axis is meaningful
    (i.e. where its own fixed dict does not pin that axis).

    HRS                                  -- all 3 panels
    HRS + aisi=AISI_FIXED                -- center, right
    HRS + sigma_w=SIGMA_W_FIXED          -- left, right
    HRS + z=Z_HALF_WIDTH_FIXED           -- left, center
    HRS + aisi + sigma_w                 -- right
    HRS + aisi + z                       -- center
    HRS + sigma_w + z                    -- left

Reads CSVs from one or more directories (typically results_6panel_v1/ and
optionally results_4panel_v1/).

Usage:
    python scripts/plot_diversity.py results_6panel_v1
    python scripts/plot_diversity.py results_6panel_v1 results_4panel_v1
    python scripts/plot_diversity.py results_6panel_v1 --output paper/fig_diversity.png
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configurable constants (match user's spec; adjust to available data if needed)
# -----------------------------------------------------------------------------

AISI_FIXED            = 0.05
SIGMA_W_FIXED         = 0.10
Z_HALF_WIDTH_FIXED    = 0.25       # IMPORTANT: not in launcher's z_width grid {0.1, 0.2, 0.3}.
                                    # Set to 0.2 to use the closest existing data.

N_TARGET   = 100
CC_TARGET  = 4
MS_TARGET  = 1


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def cfg_str(s):
    """Compact 'hom:VAL' or 'unif:LO:HI' from a JSON config dict (diversity_study output)."""
    if pd.isna(s):
        return 'NA'
    s = str(s).strip()
    if s.startswith('{'):
        try:
            d = json.loads(s)
            if d['mode'] == 'homogeneous':
                return f"hom:{d['value']}"
            return f"unif:{d['min']}:{d['max']}"
        except Exception:
            return 'NA'
    return 'NA'


def z_width(z_short):
    """Half-width of uniform z centered on 1.0; 0 for homogeneous."""
    if pd.isna(z_short):
        return np.nan
    if z_short.startswith('hom:'):
        return 0.0
    if z_short.startswith('unif:'):
        _, lo, hi = z_short.split(':')
        return round((float(hi) - float(lo)) / 2.0, 4)
    return np.nan


def load_data(dirs):
    paths = []
    for d in dirs:
        if not os.path.isdir(d):
            print(f"  warning: {d} is not a directory, skipping")
            continue
        paths.extend(sorted(glob.glob(os.path.join(d, '*.csv'))))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {dirs}")
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
        raise RuntimeError("No diversity-format CSVs found.")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['a_short'] = df['a_config'].apply(cfg_str)
    df['b_short'] = df['b_config'].apply(cfg_str)
    df['z_short'] = df['z_config'].apply(cfg_str)
    df['z_width'] = df['z_short'].apply(z_width)
    df = df[df['series'] == 'same_tech_dif_init'].copy()
    df = df[(df['n'] == N_TARGET) &
            (df['cc'] == CC_TARGET) &
            (df['max_swaps'] == MS_TARGET)].copy()
    print(f"Loaded {len(paths)} CSVs -> {len(df)} dif_init rows "
          f"(n={N_TARGET}, cc={CC_TARGET}, ms={MS_TARGET})")
    return df


# -----------------------------------------------------------------------------
# Series definitions
# -----------------------------------------------------------------------------

# Top-row (homogeneous): list of (label, a_short, b_short, z_short, color, marker)
TOP_SERIES = [
    ('CRS hom', 'hom:0.5', 'hom:1.0', 'hom:1.0', 'C0', 'o'),
    ('DRS hom', 'hom:0.5', 'hom:0.9', 'hom:1.0', 'C2', 's'),
    ('IRS hom', 'hom:0.5', 'hom:1.1', 'hom:1.0', 'C3', '^'),
]

# Bottom-row (HRS + N additional het):
# Tuple: (label, color, marker, fixed_dict, panels_list)
# `fixed_dict` keys are axis names: 'aisi_spread' / 'sigma_w' / 'z_width'.
# `panels_list` lists the panel ids ('aisi', 'sigma_w', 'z_width') where this
# series should be drawn — i.e. axes that aren't pinned by the fixed_dict.
BOTTOM_SERIES = [
    ('HRS',                                       'C0', 'o',
        {},
        ['aisi_spread', 'sigma_w', 'z_width']),
    (f'HRS + AiSi={AISI_FIXED:g}',                'C1', 's',
        {'aisi_spread': AISI_FIXED},
        ['sigma_w', 'z_width']),
    (f'HRS + $\\sigma_w$={SIGMA_W_FIXED:g}',       'C2', '^',
        {'sigma_w': SIGMA_W_FIXED},
        ['aisi_spread', 'z_width']),
    (f'HRS + $z$-width={Z_HALF_WIDTH_FIXED:g}',    'C3', 'D',
        {'z_width': Z_HALF_WIDTH_FIXED},
        ['aisi_spread', 'sigma_w']),
    (f'HRS + AiSi + $\\sigma_w$',                  'C4', 'P',
        {'aisi_spread': AISI_FIXED, 'sigma_w': SIGMA_W_FIXED},
        ['z_width']),
    (f'HRS + AiSi + $z$',                          'C5', 'X',
        {'aisi_spread': AISI_FIXED, 'z_width': Z_HALF_WIDTH_FIXED},
        ['sigma_w']),
    (f'HRS + $\\sigma_w$ + $z$',                   'C6', '*',
        {'sigma_w': SIGMA_W_FIXED, 'z_width': Z_HALF_WIDTH_FIXED},
        ['aisi_spread']),
]

# HRS economy identifiers (a, b)
HRS_A_SHORT = 'hom:0.5'
HRS_B_SHORT = 'unif:0.9:1.1'

# All three sweep axes
ALL_AXES = ['aisi_spread', 'sigma_w', 'z_width']

# Panel definitions
PANELS = [
    {'col_idx': 0, 'x_col': 'aisi_spread',
     'x_label': 'AiSi spread'},
    {'col_idx': 1, 'x_col': 'sigma_w',
     'x_label': r'Link noise $\sigma_w$'},
    {'col_idx': 2, 'x_col': 'z_width',
     'x_label': r'$z$ half-width'},
]


# -----------------------------------------------------------------------------
# Filtering / aggregation
# -----------------------------------------------------------------------------

def filter_curve(df, a_short, b_short, x_col, fixed):
    """Filter df to (a, b) and the panel's fixed-axis constraints
    (axes other than x_col are forced to 0 unless overridden in `fixed`).
    Aggregate diversity by x_col.
    """
    sub = df[(df['a_short'] == a_short) &
             (df['b_short'] == b_short)]
    panel_fixed = {ax: 0.0 for ax in ALL_AXES if ax != x_col}
    panel_fixed.update(fixed)
    for col, val in panel_fixed.items():
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

def plot_top_panel(ax, df, panel):
    """Plot homogeneous-series (CRS, DRS, IRS) for the given panel axis."""
    for label, a_short, b_short, z_short, color, marker in TOP_SERIES:
        # Top series have z=hom 1.0; we filter (a, b) and let z be implied
        # by the panel's z_width=0 default.
        g = filter_curve(df, a_short, b_short, panel['x_col'], fixed={})
        if g is None or len(g) == 0:
            continue
        ax.errorbar(g[panel['x_col']].values, g['mean'].values,
                    yerr=1.96 * g['sem'].fillna(0).values,
                    fmt=marker + '-', color=color, lw=1.6, ms=6,
                    capsize=3, label=label, alpha=0.95)


def plot_bottom_panel(ax, df, panel):
    """Plot HRS-based series for the given panel axis. Each series only
    appears if its fixed_dict allows the panel's swept axis to vary."""
    panel_axis = panel['x_col']
    for label, color, marker, fixed, panels in BOTTOM_SERIES:
        if panel_axis not in panels:
            continue
        g = filter_curve(df, HRS_A_SHORT, HRS_B_SHORT, panel_axis, fixed)
        if g is None or len(g) == 0:
            continue
        ax.errorbar(g[panel_axis].values, g['mean'].values,
                    yerr=1.96 * g['sem'].fillna(0).values,
                    fmt=marker + '-', color=color, lw=1.6, ms=6,
                    capsize=3, label=label, alpha=0.95)


def coverage_report(df):
    """Print per-(panel, series) coverage."""
    print("\n=== Coverage report ===")

    print("\n  TOP ROW (homogeneous series):")
    for panel in PANELS:
        for label, a_short, b_short, z_short, color, marker in TOP_SERIES:
            g = filter_curve(df, a_short, b_short, panel['x_col'], fixed={})
            n_pts = 0 if g is None else len(g)
            tot = 0 if g is None else int(g['count'].sum())
            print(f"    panel={panel['x_col']:13}  series={label:12}  "
                  f"x-points={n_pts:>2}  total tech-rows={tot}")

    print("\n  BOTTOM ROW (HRS-based series):")
    for panel in PANELS:
        panel_axis = panel['x_col']
        for label, color, marker, fixed, panels in BOTTOM_SERIES:
            if panel_axis not in panels:
                continue
            g = filter_curve(df, HRS_A_SHORT, HRS_B_SHORT, panel_axis, fixed)
            n_pts = 0 if g is None else len(g)
            tot = 0 if g is None else int(g['count'].sum())
            label_str = label if len(label) < 38 else label[:35] + '...'
            print(f"    panel={panel_axis:13}  series={label_str:38}  "
                  f"x-points={n_pts:>2}  total tech-rows={tot}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dirs', nargs='+',
                   help='One or more directories containing diversity CSVs.')
    p.add_argument('--output', default=None,
                   help='Output PNG path (default: <first data_dir>/figure_diversity.png)')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    df = load_data(args.data_dirs)
    coverage_report(df)

    plt.rcParams.update({
        'font.size':        11,
        'axes.titlesize':   12,
        'axes.labelsize':   11,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'legend.fontsize':   8.5,
    })

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5),
                              constrained_layout=True, sharey=True)

    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for c, panel in enumerate(PANELS):
        # Top row: homogeneous series
        ax_top = axes[0, c]
        plot_top_panel(ax_top, df, panel)
        ax_top.set_title(f"{letters[c]} Homogeneous series", loc='left')
        ax_top.set_xlabel(panel['x_label'])
        ax_top.set_ylim(-0.03, 1.03)
        ax_top.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_top.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax_top.grid(alpha=0.3)

        # Bottom row: HRS-based series
        ax_bot = axes[1, c]
        plot_bottom_panel(ax_bot, df, panel)
        ax_bot.set_title(f"{letters[3 + c]} HRS-based series", loc='left')
        ax_bot.set_xlabel(panel['x_label'])
        ax_bot.set_ylim(-0.03, 1.03)
        ax_bot.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_bot.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax_bot.grid(alpha=0.3)

    # Y-axis label only on left column
    axes[0, 0].set_ylabel('Diversity (same_tech_dif_init)')
    axes[1, 0].set_ylabel('Diversity (same_tech_dif_init)')
    for r in (0, 1):
        for c in (1, 2):
            axes[r, c].set_ylabel('')
            axes[r, c].tick_params(labelleft=False)

    # Per-row legends (top in panel (a), bottom in panel (d))
    for r, ax in [(0, axes[0, 0]), (1, axes[1, 0])]:
        h, l = ax.get_legend_handles_labels()
        if h:
            ax.legend(h, l, loc='center right', framealpha=0.95)

    # Suptitle parameters used
    fig.suptitle(
        f"AiSi$_{{\\mathrm{{fixed}}}}$={AISI_FIXED}, "
        f"$\\sigma_{{w,\\,\\mathrm{{fixed}}}}$={SIGMA_W_FIXED}, "
        f"$z_{{\\mathrm{{fixed}}}}$={Z_HALF_WIDTH_FIXED}  "
        f"|  $n$={N_TARGET}, $c={CC_TARGET}$, ms={MS_TARGET}",
        fontsize=10
    )

    out = args.output or os.path.join(args.data_dirs[0], 'figure_diversity.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nWrote {out}")


if __name__ == '__main__':
    main()
