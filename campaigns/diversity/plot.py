"""6-panel diversity figure for the manuscript (n=100, cc=4, ms=1).

Reads all CSV files in this script's directory; writes `figure_diversity.png`
in the same directory.

Layout:
                       aisi sweep         sigma_w sweep        z_width sweep
    Top (homogeneous):     (a)                (b)                  (c)
    Bottom (HRS-based):    (d)                (e)                  (f)

Top-row series:
    Homogeneous param., CRS (b=1):      a=hom 0.5, b=hom 1.0, z=hom 1.0
    Homogeneous param., DRS (b=0.9):    a=hom 0.5, b=hom 0.9, z=hom 1.0
    Homogeneous param., IRS (b=1.1):    a=hom 0.5, b=hom 1.1, z=hom 1.0

Bottom-row series (HRS = a=hom 0.5, b=unif 0.9:1.1, z=hom 1.0;
                    + 0, 1, or 2 additional fixed het sources).
Each panel's legend uses the panel-specific label form
        "HRS, <symbol_axis1>=<val>, <symbol_axis2>=<val>"
showing the non-swept fixed values.
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


# Default fixed levels for the bottom-row HRS variants
AISI_FIXED          = 0.05
SIGMA_W_FIXED       = 0.10
Z_HALF_WIDTH_FIXED  = 0.10

N_TARGET   = 100
CC_TARGET  = 4
MS_TARGET  = 1


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def cfg_str(s):
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
    df = df[(df['n'] == N_TARGET) & (df['cc'] == CC_TARGET)].copy()
    print(f"Loaded {len(paths)} CSVs -> {len(df)} dif_init rows "
          f"(n={N_TARGET}, cc={CC_TARGET}, any ms)")
    return df


# -----------------------------------------------------------------------------
# Series & panel definitions
# -----------------------------------------------------------------------------

# Top row: (label, a_short, b_short, color, marker). All shown on all 3 panels.
# Includes the bare HRS series (b uniform), distinguished from the homogeneous-b
# triplet by colour (tab:orange) and a different marker.
TOP_SERIES = [
    (r'CRS ($b_i=1$)',                  'hom:0.5', 'hom:1.0',      'C0',          'o'),
    (r'DRS ($b_i=0.9$)',                'hom:0.5', 'hom:0.9',      'C2',          's'),
    (r'IRS ($b_i=1.1$)',                'hom:0.5', 'hom:1.1',      'C3',          '^'),
    (r'HRS ($b_i\sim U[0.9,1.1]$)',     'hom:0.5', 'unif:0.9:1.1', 'tab:orange',  'D'),
]

# Bottom row: HRS economy + 1 or 2 additional fixed het sources.
# (id, color, marker, fixed_dict, panels_list)
BOTTOM_SERIES = [
    ('HRS+AiSi',    'tab:purple', 's', {'aisi_spread': AISI_FIXED},                                                 ['sigma_w', 'z_width']),
    ('HRS+sw',      'tab:brown',  '^', {'sigma_w': SIGMA_W_FIXED},                                                  ['aisi_spread', 'z_width']),
    ('HRS+z',       'tab:pink',   'D', {'z_width': Z_HALF_WIDTH_FIXED},                                             ['aisi_spread', 'sigma_w']),
    ('HRS+AiSi+sw', 'tab:gray',   'P', {'aisi_spread': AISI_FIXED, 'sigma_w': SIGMA_W_FIXED},                       ['z_width']),
    ('HRS+AiSi+z',  'tab:olive',  'X', {'aisi_spread': AISI_FIXED, 'z_width': Z_HALF_WIDTH_FIXED},                  ['sigma_w']),
    ('HRS+sw+z',    'tab:cyan',   '*', {'sigma_w': SIGMA_W_FIXED, 'z_width': Z_HALF_WIDTH_FIXED},                   ['aisi_spread']),
]

HRS_A_SHORT = 'hom:0.5'
HRS_B_SHORT = 'unif:0.9:1.1'
ALL_AXES = ['aisi_spread', 'sigma_w', 'z_width']

# Math-mode symbol per axis (used in legend labels)
AXIS_SYMBOL = {
    'aisi_spread': r'$\Delta_A$',
    'sigma_w':     r'$\sigma_w$',
    'z_width':     r'$\Delta_z$',
}


def panel_legend_label(fixed, panel_axis):
    """Build a panel-specific legend label of the form
        '<sym1>=<v1>, <sym2>=<v2>'
    where sym1/sym2 are the two non-swept axes (HRS is implicit -- the
    legend title says so)."""
    parts = []
    for ax in ALL_AXES:
        if ax == panel_axis:
            continue
        val = fixed.get(ax, 0.0)
        parts.append(f"{AXIS_SYMBOL[ax]}={val:g}")
    return ', '.join(parts)


# Three-line x-axis labels for the bottom row.
XLABELS = {
    'aisi_spread': '\n'.join([
        r'Heterogeneity $\Delta_A$ in',
        r'supplier-combination productivity $A_i(\mathcal{S}_i)$',
        r'$A_i(\mathcal{S}_i)\sim U[1-\Delta_A,\, 1+\Delta_A]$',
    ]),
    'sigma_w': '\n'.join([
        r'Heterogeneity $\sigma_w$ in',
        r'link productivity $W_{ij}$',
        r'$W_{ij}\sim \mathcal{N}(1/c_i,\, \sigma_w^2)$',
    ]),
    'z_width': '\n'.join([
        r'Heterogeneity $\Delta_z$ in',
        r'firm productivity $z_i$',
        r'$z_i\sim U[1-\Delta_z,\, 1+\Delta_z]$',
    ]),
}

PANELS = [
    {'col_idx': 0, 'x_col': 'aisi_spread'},
    {'col_idx': 1, 'x_col': 'sigma_w'},
    {'col_idx': 2, 'x_col': 'z_width'},
]


# -----------------------------------------------------------------------------
# Filtering / aggregation
# -----------------------------------------------------------------------------

def filter_curve(df, a_short, b_short, x_col, fixed, ms=MS_TARGET):
    sub = df[(df['a_short'] == a_short) &
             (df['b_short'] == b_short) &
             (df['max_swaps'] == ms)]
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

def plot_top_panel(ax, df, panel, ms=MS_TARGET):
    for label, a_short, b_short, color, marker in TOP_SERIES:
        g = filter_curve(df, a_short, b_short, panel['x_col'], fixed={}, ms=ms)
        if g is None or len(g) == 0:
            continue
        ax.errorbar(g[panel['x_col']].values, g['mean'].values,
                    yerr=1.96 * g['sem'].fillna(0).values,
                    fmt=marker + '-', color=color, lw=2.5, ms=10,
                    capsize=5, label=label, alpha=0.95)


def plot_bottom_panel(ax, df, panel, ms=MS_TARGET):
    panel_axis = panel['x_col']
    for sid, color, marker, fixed, panels in BOTTOM_SERIES:
        if panel_axis not in panels:
            continue
        g = filter_curve(df, HRS_A_SHORT, HRS_B_SHORT, panel_axis, fixed, ms=ms)
        if g is None or len(g) == 0:
            continue
        label = panel_legend_label(fixed, panel_axis)
        ax.errorbar(g[panel_axis].values, g['mean'].values,
                    yerr=1.96 * g['sem'].fillna(0).values,
                    fmt=marker + '-', color=color, lw=2.5, ms=10,
                    capsize=5, label=label, alpha=0.95)


def coverage_report(df):
    print("\n=== Coverage report ===")
    print("\n  TOP ROW:")
    for panel in PANELS:
        for label, a, b, *_ in TOP_SERIES:
            g = filter_curve(df, a, b, panel['x_col'], {})
            n_pts = 0 if g is None else len(g)
            tot = 0 if g is None else int(g['count'].sum())
            print(f"    panel={panel['x_col']:13}  series={label[:38]:38}  "
                  f"x-points={n_pts:>2}  total rows={tot}")
    print("\n  BOTTOM ROW:")
    for panel in PANELS:
        panel_axis = panel['x_col']
        for sid, color, marker, fixed, panels in BOTTOM_SERIES:
            if panel_axis not in panels:
                continue
            g = filter_curve(df, HRS_A_SHORT, HRS_B_SHORT, panel_axis, fixed)
            label = panel_legend_label(fixed, panel_axis)
            n_pts = 0 if g is None else len(g)
            tot = 0 if g is None else int(g['count'].sum())
            print(f"    panel={panel_axis:13}  series={label[:38]:38}  "
                  f"x-points={n_pts:>2}  total rows={tot}")


# Default legend positions for the ms=1 figure (chosen to avoid the curves).
DEFAULT_LEGEND_LOCS = {
    'a': 'lower center',   # (a) top-left
    'd': 'lower center',   # (d) bottom-left
    'e': 'lower center',   # (e) bottom-center
    'f': 'center right',   # (f) bottom-right
}


def make_6panel_figure(df, save_path, ms, legend_locs=None):
    """Build the 6-panel diversity figure at the requested max_swaps level.

    legend_locs: optional dict overriding the per-panel matplotlib loc string
        for panels 'a', 'd', 'e', 'f'. Missing keys fall back to
        DEFAULT_LEGEND_LOCS.
    """
    legend_locs = {**DEFAULT_LEGEND_LOCS, **(legend_locs or {})}

    # Tall figure to accommodate 3-line x-axis labels + 4-row legends.
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 10.0), sharey=True)

    letters = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    for c, panel in enumerate(PANELS):
        # --- Top row ---
        ax_top = axes[0, c]
        plot_top_panel(ax_top, df, panel, ms=ms)
        ax_top.set_title(letters[c], loc='left')
        ax_top.set_xlabel('')
        ax_top.tick_params(labelbottom=False)
        ax_top.set_ylim(-0.03, 1.03)
        ax_top.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_top.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax_top.grid(alpha=0.3)

        # --- Bottom row ---
        ax_bot = axes[1, c]
        plot_bottom_panel(ax_bot, df, panel, ms=ms)
        ax_bot.set_title(letters[3 + c], loc='left')
        ax_bot.set_xlabel(XLABELS[panel['x_col']])
        ax_bot.set_ylim(-0.03, 1.03)
        ax_bot.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax_bot.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax_bot.grid(alpha=0.3)

    # Y-axis label only on left column.
    axes[0, 0].set_ylabel('Diversity of final network (%)')
    axes[1, 0].set_ylabel('Diversity of final network (%)')
    for r in (0, 1):
        for c in (1, 2):
            axes[r, c].set_ylabel('')
            axes[r, c].tick_params(labelleft=False)

    # Legends (positions overrideable via legend_locs).
    h, l = axes[0, 0].get_legend_handles_labels()
    if h:
        axes[0, 0].legend(h, l, loc=legend_locs['a'],
                          title=r'Homo. prod.',
                          framealpha=0.95)
    h, l = axes[1, 0].get_legend_handles_labels()
    if h:
        axes[1, 0].legend(h, l, loc=legend_locs['d'],
                          title='Hetero. prod. and HRS',
                          framealpha=0.95)
    h, l = axes[1, 1].get_legend_handles_labels()
    if h:
        axes[1, 1].legend(h, l, loc=legend_locs['e'], framealpha=0.95)
    h, l = axes[1, 2].get_legend_handles_labels()
    if h:
        axes[1, 2].legend(h, l, loc=legend_locs['f'], framealpha=0.95)

    # Reserve space for 3-line x-labels and larger fonts; widen inter-panel gap.
    fig.subplots_adjust(left=0.07, right=0.99, top=0.97, bottom=0.17,
                        wspace=0.10, hspace=0.10)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


def main():
    # Tracked code in campaigns/diversity/; CSVs + output PNGs live in
    # results/diversity/ (gitignored).
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    default_data_dir = os.path.join(repo_root, 'results', 'diversity')

    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=default_data_dir,
                   help=f'Directory containing diversity CSVs. '
                        f'Default: {default_data_dir}.')
    p.add_argument('--output', default=None,
                   help='Output PNG path for the ms=1 figure '
                        '(default: <data_dir>/figure_diversity.png).')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    df = load_data([args.data_dir])
    coverage_report(df)

    # Font sizes sized for ~2.15x scaling: the rendered figure is 13.5 in
    # wide; included at \textwidth (~6.27 in on A4 with 1-in margins) it
    # scales by ~0.46.  Target on-page sizes (body 8 pt, ticks 6.5 pt) are
    # standard for full-width manuscript figures.
    FONT_SCALE = 2.15
    plt.rcParams.update({
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
    })

    # --- ms=1 (main figure) -------------------------------------------------
    out_ms1 = args.output or os.path.join(args.data_dir, 'figure_diversity.png')
    make_6panel_figure(df, out_ms1, ms=MS_TARGET)

    # --- ms=4 (appendix companion) ------------------------------------------
    # At ms=4 the homogeneous curves saturate at the top of panel (a), and the
    # bottom-row curves leave the top of (d)/(e)/(f) free, so the legends move:
    #   (a) -> middle center;  (d, e, f) -> upper center.
    out_ms4 = os.path.join(args.data_dir, 'figure_diversity_ms4.png')
    ms4_legend_locs = {
        'a': 'center',
        'd': 'upper center',
        'e': 'upper center',
        'f': 'upper center',
    }
    make_6panel_figure(df, out_ms4, ms=4, legend_locs=ms4_legend_locs)


if __name__ == '__main__':
    main()
