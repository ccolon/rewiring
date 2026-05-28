"""3-panel figure for the switching-cost (chi) experiment.

Reads trial-level CSVs from results/switchcosts/ (or a user-supplied directory)
and produces the figure described in the manuscript appendix:

    (a) Configuration diversity nu_P     vs chi
    (b) Mean terminal cost gap theta     vs chi   (theta_static)
    (c) Per-firm rewiring count F        vs chi   (total_rewirings / n)

Two lines per panel (one per parameter point in {P2, P4} -- labels match
welfare_dispersion_study.py). Markers at each chi value; bars are 95%
normal-approx confidence intervals.

Per-cell metric definitions:
    nu_P : per (point, chi, tech_seed) group the trials by their shared tech
           matrix, count unique final supplier-set configurations (via
           final_config_hash), divide by (n_inits - 1). Mean across tech_seeds
           is the cell value; CI is computed across tech_seeds.
    theta: mean of trial-level theta_static (Option-B static gap against the
           unconstrained best deviation).
    F    : mean of trial-level total_rewirings / n.

Usage:
    python campaigns/switchcosts/plot.py results/switchcosts
    python campaigns/switchcosts/plot.py results/switchcosts --output paper/fig_switchcost.png
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Plot style: point label -> (display label, color).
POINT_STYLE = {
    'P2': ('P2  CRS, $\\kappa=1$, $\\Delta_A=0.05$',  'C0'),
    'P3': ('P3  HRS, $\\kappa=c\'$, no dispersion',   'C2'),
    'P4': ('P4  HRS, $\\kappa=1$, compounded',        'C1'),
}


def load(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {data_dir!r}")
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        # Switchcost CSVs have a 'chi' column and a 'theta_static' column.
        if 'chi' not in d.columns or 'theta_static' not in d.columns:
            continue
        dfs.append(d)
    if not dfs:
        raise RuntimeError(f"No switchcost-format CSVs in {data_dir}")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"Loaded {len(paths)} CSV(s) -> {len(df)} rows")
    return df


def nu_p_per_tech(group, n_per_tech_expected=None):
    """Diversity within a single (point, chi, tech_seed) group.

    (#unique configs - 1) / (K - 1), where K = len(group). Returns NaN if
    K <= 1 (under-sampled).
    """
    k = len(group)
    if k <= 1:
        return float('nan')
    hashes = group['final_config_hash'].unique()
    return (len(hashes) - 1) / (k - 1)


def aggregate_cell(df_cell, n_firms):
    """Per-(point, chi) cell, compute three metrics with 95% CIs."""
    out = {}
    # (a) nu_P: compute per tech_seed, then mean +/- CI across techs.
    # Pass only the columns we need to silence the include_groups DeprecationWarning.
    by_tech = (df_cell[['tech_seed', 'final_config_hash']]
               .groupby('tech_seed', group_keys=False)
               .apply(nu_p_per_tech))
    by_tech = by_tech.dropna()
    out['nu_p_n']   = len(by_tech)
    out['nu_p_mean'] = float(by_tech.mean()) if len(by_tech) else float('nan')
    out['nu_p_ci']   = (1.96 * float(by_tech.std(ddof=1)) / np.sqrt(len(by_tech))
                        if len(by_tech) > 1 else float('nan'))

    # (b) theta: mean of theta_static across all trials.
    th = df_cell['theta_static'].astype(float)
    out['theta_n']    = len(th)
    out['theta_mean'] = float(th.mean()) if len(th) else float('nan')
    out['theta_ci']   = (1.96 * float(th.std(ddof=1)) / np.sqrt(len(th))
                         if len(th) > 1 else float('nan'))

    # (c) F: per-firm rewirings = total_rewirings / n.
    f = df_cell['total_rewirings'].astype(float) / float(n_firms)
    out['F_n']    = len(f)
    out['F_mean'] = float(f.mean()) if len(f) else float('nan')
    out['F_ci']   = (1.96 * float(f.std(ddof=1)) / np.sqrt(len(f))
                     if len(f) > 1 else float('nan'))
    return out


def build_summary(df, n_firms):
    """Return a long-format DataFrame: one row per (point, chi)."""
    rows = []
    for (point, chi), grp in df.groupby(['point', 'chi']):
        agg = aggregate_cell(grp, n_firms)
        rows.append({'point': point, 'chi': float(chi), **agg})
    return pd.DataFrame(rows).sort_values(['point', 'chi']).reset_index(drop=True)


def plot_panel(ax, summary, mean_col, ci_col, ylabel, title,
               points_to_plot, use_pct=False, log_y=False, chi_grid=None):
    for point in points_to_plot:
        sub = summary[summary['point'] == point].sort_values('chi')
        if sub.empty:
            continue
        label, color = POINT_STYLE.get(point, (point, None))
        x = sub['chi'].to_numpy()
        y = sub[mean_col].to_numpy()
        e = sub[ci_col].to_numpy()
        if use_pct:
            y = y * 100.0
            e = e * 100.0
        ax.errorbar(x, y, yerr=e, fmt='o-', color=color,
                    lw=1.8, ms=7, capsize=3, label=label)
    ax.set_xlabel(r'$\chi$ (per-switch hurdle)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    ax.grid(alpha=0.3)
    if chi_grid is not None and len(chi_grid):
        ax.set_xticks(chi_grid)
        ax.set_xticklabels([f'{c:g}' for c in chi_grid], rotation=30, ha='right')
    if log_y:
        ax.set_yscale('log')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_dir',
                   help='Directory with switchcost trial CSVs (e.g. results/switchcosts).')
    p.add_argument('--output', default=None,
                   help='Output figure path. Defaults to <data_dir>/fig_switchcost.png.')
    p.add_argument('--n_firms', type=int, default=100,
                   help='Number of firms (for the F = total_rewirings / n metric). '
                        'Default 100.')
    p.add_argument('--points', default='P2,P4',
                   help='Comma-separated subset of points to plot (default: P2,P4).')
    p.add_argument('--log_x', action='store_true',
                   help='Use symlog x-axis (helpful when chi spans >2 orders of '
                        'magnitude with chi=0 at the left).')
    args = p.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data_dir, 'fig_switchcost.png')

    df = load(args.data_dir)
    points_to_plot = args.points.split(',')

    summary = build_summary(df, args.n_firms)
    print('\n=== Per-cell summary ===')
    with pd.option_context('display.max_rows', 50,
                           'display.float_format', '{:.4g}'.format):
        print(summary[['point', 'chi', 'nu_p_mean', 'nu_p_ci',
                       'theta_mean', 'theta_ci', 'F_mean', 'F_ci',
                       'nu_p_n', 'theta_n']])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    chi_grid = sorted(summary['chi'].unique())

    plot_panel(axes[0], summary, 'nu_p_mean', 'nu_p_ci',
               ylabel=r'Configuration diversity $\nu_\mathcal{P}$',
               title='(a)', points_to_plot=points_to_plot, chi_grid=chi_grid)
    plot_panel(axes[1], summary, 'theta_mean', 'theta_ci',
               ylabel=r'Terminal cost gap $\theta$ (%)',
               title='(b)', points_to_plot=points_to_plot, use_pct=True,
               chi_grid=chi_grid)
    plot_panel(axes[2], summary, 'F_mean', 'F_ci',
               ylabel=r'Per-firm rewiring count $F$',
               title='(c)', points_to_plot=points_to_plot, chi_grid=chi_grid)

    if args.log_x:
        for ax in axes:
            # Use symlog to keep chi=0 visible.
            ax.set_xscale('symlog', linthresh=5e-4)

    # Shared legend below the panels.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=10)

    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"\nFigure -> {args.output}")
    pdf_out = os.path.splitext(args.output)[0] + '.pdf'
    fig.savefig(pdf_out, bbox_inches='tight')
    print(f"Figure -> {pdf_out}")


if __name__ == '__main__':
    main()
