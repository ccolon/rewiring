"""Render the 4-panel manuscript figure from `results_4panel_v1/`.

Layout (Y axis is diversity = same_tech_dif_init):

    +--------------------+--------------------+
    | ms=1   x=aisi      | ms=1   x=sigma_w   |   (sigma_w=0 fixed)   (aisi=0 fixed)
    +--------------------+--------------------+
    | ms=cc=4 x=aisi     | ms=cc=4 x=sigma_w  |
    +--------------------+--------------------+

Series shown on each panel (n=100, cc=4 unless noted):
    1.  fully hom CRS:        a=hom 0.5,    b=hom 1.0,     z=hom 1.0
    2.  hom + b uniform:      a=hom 0.5,    b=unif 0.9:1.1, z=hom 1.0
    3.  realistic:            a=unif 0.4:0.6, b=unif 0.9:1.1, z=unif 0.9:1.1
    4.  realistic + wide z:   a=unif 0.4:0.6, b=unif 0.9:1.1, z=unif 0.5:1.5
    5.  realistic, cc=2 (top row only):  same as 3 with cc=2
    6.  realistic, n=200:     same as 3 with n=200

Reads all `4panel_*.csv` files in the input dir, concatenates, filters to
same_tech_dif_init rows, and aggregates per cell (mean ± 1.96·SEM across
tech matrices).

Usage:
    python scripts/plot_4panel.py                                    # default dir
    python scripts/plot_4panel.py results_4panel_v1
    python scripts/plot_4panel.py results_4panel_v1 --output fig.png
"""
import argparse
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# (id, label, match_keys, color, marker, linestyle)
SERIES_DEFS = [
    ('s1', '1: hom CRS',
     dict(a='hom:0.5',      b='hom:1.0',      z='hom:1.0',      n=100, cc=4),
     'C0', 'o', '-'),
    ('s2', '2: hom + b unif',
     dict(a='hom:0.5',      b='unif:0.9:1.1', z='hom:1.0',      n=100, cc=4),
     'C1', 's', '-'),
    ('s3', '3: realistic',
     dict(a='unif:0.4:0.6', b='unif:0.9:1.1', z='unif:0.9:1.1', n=100, cc=4),
     'C2', '^', '-'),
    ('s4', '4: realistic + wide z',
     dict(a='unif:0.4:0.6', b='unif:0.9:1.1', z='unif:0.5:1.5', n=100, cc=4),
     'C3', 'D', '-'),
    ('s5', '5: realistic, cc=2',
     dict(a='unif:0.4:0.6', b='unif:0.9:1.1', z='unif:0.9:1.1', n=100, cc=2),
     'C4', 'P', '--'),
    ('s6', '6: realistic, n=200',
     dict(a='unif:0.4:0.6', b='unif:0.9:1.1', z='unif:0.9:1.1', n=200, cc=4),
     'C5', 'X', '--'),
]


def cfg_str(s):
    """Compact 'hom:VAL' or 'unif:LO:HI' from a JSON config string."""
    try:
        d = json.loads(s)
        if d['mode'] == 'homogeneous':
            return f"hom:{d['value']}"
        return f"unif:{d['min']}:{d['max']}"
    except Exception:
        return 'NA'


def load_data(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, '4panel_*.csv')))
    if not paths:
        raise FileNotFoundError(
            f"No 4panel_*.csv files found in {data_dir!r}. "
            f"Did the launcher's --output path land somewhere else?"
        )
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"  skipping {os.path.basename(p)}: {e}", file=sys.stderr)
            continue
        if len(d):
            dfs.append(d)
    if not dfs:
        raise RuntimeError(f"All CSVs in {data_dir} were empty or unreadable.")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['a_short'] = df['a_config'].apply(cfg_str)
    df['b_short'] = df['b_config'].apply(cfg_str)
    df['z_short'] = df['z_config'].apply(cfg_str)
    df = df[df['series'] == 'same_tech_dif_init'].copy()
    print(f"Loaded {len(paths)} CSVs -> {len(df)} dif_init rows")
    return df


def select_series(df, keys):
    return df[(df['a_short'] == keys['a']) & (df['b_short'] == keys['b']) &
              (df['z_short'] == keys['z']) & (df['n'] == keys['n']) &
              (df['cc'] == keys['cc'])]


def plot_panel(ax, df, max_swaps, sweep_axis, title, plot_s5=False):
    """sweep_axis: 'aisi_spread' (the other = sigma_w fixed at 0) or 'sigma_w'."""
    fixed_col = 'sigma_w' if sweep_axis == 'aisi_spread' else 'aisi_spread'
    plotted_anything = False

    for sid, label, keys, color, marker, ls in SERIES_DEFS:
        if sid == 's5' and not plot_s5:
            continue
        sub = select_series(df, keys)
        sub = sub[(sub['max_swaps'] == max_swaps) & (sub[fixed_col] == 0.0)]
        if len(sub) == 0:
            continue
        g = (sub.groupby(sweep_axis)['diversity']
                 .agg(['mean', 'sem', 'count'])
                 .sort_index())
        x = g.index.values.astype(float)
        y = g['mean'].values
        yerr = (1.96 * g['sem'].fillna(0.0)).values
        ax.errorbar(x, y, yerr=yerr, fmt=marker, ls=ls, color=color,
                    markersize=7, lw=1.6, capsize=3, label=label, alpha=0.9)
        plotted_anything = True

    if sweep_axis == 'aisi_spread':
        ax.set_xlabel(r'AiSi spread $\rho_{\mathrm{AiSi}}$  ($\sigma_w=0$)')
    else:
        ax.set_xlabel(r'link noise $\sigma_w$  (AiSi=0)')
    ax.set_ylabel('diversity  (same_tech_dif_init)')
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    if not plotted_anything:
        ax.text(0.5, 0.5, '(no data)', transform=ax.transAxes,
                ha='center', va='center', color='gray', fontsize=12)


def coverage_report(df):
    """Print which (series × ms × axis × value) cells exist."""
    print("\n=== Coverage report (per series, ms, sweep cell) ===")
    for sid, label, keys, *_ in SERIES_DEFS:
        sub = select_series(df, keys)
        if len(sub) == 0:
            print(f"  {sid} {label}: NO DATA")
            continue
        for ms in [1, 4, 2]:  # cc=2 => ms=2 possible too, even though we don't plot it
            sub_ms = sub[sub['max_swaps'] == ms]
            if len(sub_ms) == 0:
                continue
            cells = (sub_ms.groupby(['aisi_spread', 'sigma_w'])
                            ['diversity'].agg(['mean', 'count']))
            n_cells = len(cells)
            n_tech_total = cells['count'].sum()
            cell_str = ', '.join(
                f"(a={a:g},sw={w:g}):{c}" for (a, w), c in cells['count'].items()
            )
            print(f"  {sid} ms={ms}: {n_cells} cells, {n_tech_total} tech-rows  -> {cell_str}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('data_dir', nargs='?',
                        default='results_4panel_v1',
                        help='Directory containing 4panel_*.csv files '
                             '(default: results_4panel_v1).')
    parser.add_argument('--output', default=None,
                        help='Output PNG path (default: <data_dir>/figure_4panel.png).')
    args = parser.parse_args()

    df = load_data(args.data_dir)
    coverage_report(df)

    out_path = args.output or os.path.join(args.data_dir, 'figure_4panel.png')

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), constrained_layout=True)

    plot_panel(axes[0, 0], df, max_swaps=1, sweep_axis='aisi_spread',
               title=r'$ms=1$,  $\sigma_w=0$', plot_s5=True)
    plot_panel(axes[0, 1], df, max_swaps=1, sweep_axis='sigma_w',
               title=r'$ms=1$,  AiSi$=0$', plot_s5=True)
    plot_panel(axes[1, 0], df, max_swaps=4, sweep_axis='aisi_spread',
               title=r'$ms=cc=4$,  $\sigma_w=0$', plot_s5=False)
    plot_panel(axes[1, 1], df, max_swaps=4, sweep_axis='sigma_w',
               title=r'$ms=cc=4$,  AiSi$=0$', plot_s5=False)

    # Single legend below the grid
    handles_labels = []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in [x[1] for x in handles_labels]:
                handles_labels.append((h, l))
    if handles_labels:
        h, l = zip(*handles_labels)
        fig.legend(h, l, loc='lower center', ncol=3, fontsize=9,
                   bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        'Diversity (same_tech_dif_init) vs heterogeneity, '
        'two action regimes  |  n=100 cc=4 unless noted',
        fontsize=12,
    )
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
