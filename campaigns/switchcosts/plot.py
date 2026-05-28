"""Switching-cost (chi) figure -- main 3 panels + 2 diagnostic panels.

Reads trial-level CSVs from results/switchcosts/ (or a user-supplied directory).

Main panels (manuscript appendix fig:switchcost):
    (a) Configuration diversity nu_P     vs chi
    (b) Mean terminal cost gap theta     vs chi   (theta_static)
    (c) Per-firm rewiring count F        vs chi   (total_rewirings / n)

Diagnostic panels (reuse the same data; no new sims needed):
    (d) Termination breakdown            vs chi   (frac converged + frac cycled)
    (e) Matched-pair retention           vs chi   (fraction of trials whose
                                                   final configuration equals
                                                   the chi=0 baseline trial at
                                                   the same (tech_seed,
                                                   init_seed); anchored at 1
                                                   for chi=0).

The 6th axis carries a small sample-size table per cell.

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
    frac_converged : fraction of trials with converged == 1 (period-1).
    frac_cycled    : fraction with cycle_period in {2, ..., MAX_CYCLE_PERIOD}.
    M    : matched-pair retention. For each (tech_seed, init_seed) pair, look
           up the chi=0 final_config_hash; for a trial at (point, chi, tech,
           init), M_i = 1 if its hash equals the chi=0 baseline hash else 0.
           Reported as the mean of M_i over all available pairs. M(chi=0)=1
           by construction.

Usage:
    python campaigns/switchcosts/plot.py results/switchcosts
    python campaigns/switchcosts/plot.py results/switchcosts --output paper/fig_switchcost.png
    python campaigns/switchcosts/plot.py results/switchcosts --log_x --main_only
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


# =============================================================================
# Data loading
# =============================================================================

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
        if 'chi' not in d.columns or 'theta_static' not in d.columns:
            continue
        dfs.append(d)
    if not dfs:
        raise RuntimeError(f"No switchcost-format CSVs in {data_dir}")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"Loaded {len(paths)} CSV(s) -> {len(df)} rows")
    return df


# =============================================================================
# Per-cell aggregation: nu_P, theta, F, convergence breakdown
# =============================================================================

def nu_p_per_tech(group):
    """Diversity within a single (point, chi, tech_seed) group:
       (#unique configs - 1) / (K - 1). NaN if K <= 1."""
    k = len(group)
    if k <= 1:
        return float('nan')
    hashes = group['final_config_hash'].unique()
    return (len(hashes) - 1) / (k - 1)


def _bin_ci(p, n):
    """95% normal-approx CI half-width for a binomial proportion."""
    if n <= 1:
        return float('nan')
    return 1.96 * np.sqrt(p * (1.0 - p) / n)


def aggregate_cell(df_cell, n_firms):
    """Per-(point, chi) cell, compute the five base metrics with 95% CIs."""
    out = {}
    # (a) nu_P: per tech_seed then mean +/- CI across techs.
    by_tech = (df_cell[['tech_seed', 'final_config_hash']]
               .groupby('tech_seed', group_keys=False)
               .apply(nu_p_per_tech))
    by_tech = by_tech.dropna()
    out['nu_p_n']   = len(by_tech)
    out['nu_p_mean'] = float(by_tech.mean()) if len(by_tech) else float('nan')
    out['nu_p_ci']   = (1.96 * float(by_tech.std(ddof=1)) / np.sqrt(len(by_tech))
                        if len(by_tech) > 1 else float('nan'))

    # (b) theta: mean theta_static across all trials.
    th = df_cell['theta_static'].astype(float)
    out['theta_n']    = len(th)
    out['theta_mean'] = float(th.mean()) if len(th) else float('nan')
    out['theta_ci']   = (1.96 * float(th.std(ddof=1)) / np.sqrt(len(th))
                         if len(th) > 1 else float('nan'))

    # (c) F: per-firm rewirings.
    f = df_cell['total_rewirings'].astype(float) / float(n_firms)
    out['F_n']    = len(f)
    out['F_mean'] = float(f.mean()) if len(f) else float('nan')
    out['F_ci']   = (1.96 * float(f.std(ddof=1)) / np.sqrt(len(f))
                     if len(f) > 1 else float('nan'))

    # (d) Termination breakdown.
    conv = df_cell['converged'].astype(int).to_numpy()
    cp = pd.to_numeric(df_cell['cycle_period'], errors='coerce').to_numpy()
    n_trials = len(df_cell)
    p_fix = float(conv.mean()) if n_trials else float('nan')
    cycled_mask = (~np.isnan(cp)) & (cp >= 2)
    p_cyc = float(cycled_mask.mean()) if n_trials else float('nan')
    out['frac_conv_n']    = n_trials
    out['frac_conv_mean'] = p_fix
    out['frac_conv_ci']   = _bin_ci(p_fix, n_trials)
    out['frac_cyc_mean']  = p_cyc
    out['frac_cyc_ci']    = _bin_ci(p_cyc, n_trials)
    return out


def build_summary(df, n_firms):
    """One row per (point, chi) with all aggregated metrics."""
    rows = []
    for (point, chi), grp in df.groupby(['point', 'chi']):
        rows.append({'point': point, 'chi': float(chi),
                     **aggregate_cell(grp, n_firms)})
    return pd.DataFrame(rows).sort_values(['point', 'chi']).reset_index(drop=True)


# =============================================================================
# Matched-pair retention vs chi=0 baseline
# =============================================================================

def matched_pair_summary(df):
    """Per (point, chi), fraction of (tech_seed, init_seed) pairs whose final
    configuration matches the chi=0 baseline trial at the same pair.

    chi=0 is included and trivially evaluates to 1.0.
    """
    rows = []
    for point, sub in df.groupby('point'):
        baseline = (sub[np.isclose(sub['chi'], 0.0)]
                    .set_index(['tech_seed', 'init_seed'])
                    ['final_config_hash']
                    .to_dict())
        if not baseline:
            print(f"  [warn] {point}: no chi=0 trials found; "
                  f"skipping matched-pair metric")
            continue
        for chi, chi_sub in sub.groupby('chi'):
            matches = []
            for tech_seed, init_seed, h in zip(
                    chi_sub['tech_seed'].to_numpy(),
                    chi_sub['init_seed'].to_numpy(),
                    chi_sub['final_config_hash'].to_numpy()):
                key = (tech_seed, init_seed)
                if key in baseline:
                    matches.append(int(h == baseline[key]))
            if not matches:
                continue
            m = np.asarray(matches, dtype=float)
            mean = float(m.mean())
            rows.append({
                'point':  point,
                'chi':    float(chi),
                'M_mean': mean,
                'M_ci':   _bin_ci(mean, len(m)),
                'M_n':    int(len(m)),
            })
    return pd.DataFrame(rows).sort_values(['point', 'chi']).reset_index(drop=True)


# =============================================================================
# Panel plotting helpers
# =============================================================================

def plot_panel(ax, summary, mean_col, ci_col, ylabel, title,
               points_to_plot, use_pct=False, log_y=False, chi_grid=None,
               extra=None):
    """Generic single-metric panel. `extra` is an optional list of
    (mean_col, ci_col, label_suffix, linestyle) to overlay on the same axis
    (e.g. frac_cycled on top of frac_converged)."""
    for point in points_to_plot:
        sub = summary[summary['point'] == point].sort_values('chi')
        if sub.empty:
            continue
        label, color = POINT_STYLE.get(point, (point, None))
        x = sub['chi'].to_numpy()
        y = sub[mean_col].to_numpy()
        e = sub[ci_col].to_numpy()
        if use_pct:
            y, e = y * 100.0, e * 100.0
        ax.errorbar(x, y, yerr=e, fmt='o-', color=color,
                    lw=1.8, ms=7, capsize=3, label=label)

        if extra is not None:
            for em_col, eci_col, lbl_suffix, ls in extra:
                if em_col not in sub.columns:
                    continue
                y2 = sub[em_col].to_numpy()
                e2 = sub[eci_col].to_numpy() if eci_col in sub.columns else None
                if use_pct:
                    y2 = y2 * 100.0
                    if e2 is not None:
                        e2 = e2 * 100.0
                ax.errorbar(x, y2, yerr=e2, fmt='x', color=color, ls=ls,
                            lw=1.2, ms=6, capsize=2, alpha=0.7,
                            label=f'{label} -- {lbl_suffix}')

    ax.set_xlabel(r'$\chi$ (per-switch hurdle)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc='left')
    ax.grid(alpha=0.3)
    if chi_grid is not None and len(chi_grid):
        ax.set_xticks(chi_grid)
        ax.set_xticklabels([f'{c:g}' for c in chi_grid],
                           rotation=30, ha='right')
    if log_y:
        ax.set_yscale('log')


def render_info_panel(ax, summary, mp_summary, points_to_plot):
    """6th cell: small table of sample sizes per (point, chi) for diagnostics."""
    ax.set_axis_off()
    lines = ["Sample sizes per cell (n_trials | n_techs):"]
    by_pc = summary.set_index(['point', 'chi'])
    chis = sorted(summary['chi'].unique())
    header = "  chi      " + "  ".join(f"{p:>10s}" for p in points_to_plot)
    lines.append(header)
    for chi in chis:
        cells = []
        for p in points_to_plot:
            if (p, chi) in by_pc.index:
                row = by_pc.loc[(p, chi)]
                cells.append(f"{int(row['theta_n']):>4d}|{int(row['nu_p_n']):>3d}")
            else:
                cells.append("   - ")
        lines.append(f"  {chi:>7.4g}  " + "  ".join(f"{c:>10s}" for c in cells))
    if not mp_summary.empty:
        mp_by = mp_summary.set_index(['point', 'chi'])
        lines.append("")
        lines.append("Matched-pair n per (point, chi):")
        lines.append("  chi      " + "  ".join(f"{p:>10s}" for p in points_to_plot))
        for chi in chis:
            cells = []
            for p in points_to_plot:
                if (p, chi) in mp_by.index:
                    cells.append(f"{int(mp_by.loc[(p, chi), 'M_n']):>10d}")
                else:
                    cells.append(f"{'-':>10s}")
            lines.append(f"  {chi:>7.4g}  " + "  ".join(cells))
    ax.text(0.0, 1.0, "\n".join(lines),
            family='monospace', fontsize=8,
            transform=ax.transAxes, va='top', ha='left')


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_dir',
                   help='Directory with switchcost trial CSVs (e.g. results/switchcosts).')
    p.add_argument('--output', default=None,
                   help='Output figure path. Defaults to <data_dir>/fig_switchcost.png.')
    p.add_argument('--n_firms', type=int, default=100,
                   help='Number of firms (for F = total_rewirings / n). Default 100.')
    p.add_argument('--points', default='P2,P4',
                   help='Comma-separated subset of points to plot (default: P2,P4).')
    p.add_argument('--log_x', action='store_true',
                   help='Use symlog x-axis (helpful when chi spans >2 orders of '
                        'magnitude with chi=0 at the left).')
    p.add_argument('--main_only', action='store_true',
                   help='Render only the 3-panel main figure (a, b, c). '
                        'Default also renders the 2 diagnostic panels (d, e).')
    args = p.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data_dir, 'fig_switchcost.png')

    df = load(args.data_dir)
    points_to_plot = args.points.split(',')

    summary = build_summary(df, args.n_firms)
    mp_summary = matched_pair_summary(df) if not args.main_only else pd.DataFrame()

    print('\n=== Per-cell summary (main metrics) ===')
    with pd.option_context('display.max_rows', 50,
                           'display.float_format', '{:.4g}'.format):
        print(summary[['point', 'chi', 'nu_p_mean', 'nu_p_ci',
                       'theta_mean', 'theta_ci', 'F_mean', 'F_ci',
                       'frac_conv_mean', 'frac_cyc_mean',
                       'nu_p_n', 'theta_n']])
    if not mp_summary.empty:
        print('\n=== Matched-pair retention vs chi=0 ===')
        with pd.option_context('display.max_rows', 50,
                               'display.float_format', '{:.4g}'.format):
            print(mp_summary)

    chi_grid = sorted(summary['chi'].unique())

    if args.main_only:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5),
                                 constrained_layout=True)
        main_axes = axes
    else:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                                 constrained_layout=True)
        main_axes = axes[0]
        diag_axes = axes[1]

    plot_panel(main_axes[0], summary, 'nu_p_mean', 'nu_p_ci',
               ylabel=r'Configuration diversity $\nu_\mathcal{P}$',
               title='(a)', points_to_plot=points_to_plot, chi_grid=chi_grid)
    plot_panel(main_axes[1], summary, 'theta_mean', 'theta_ci',
               ylabel=r'Terminal cost gap $\theta$ (%)',
               title='(b)', points_to_plot=points_to_plot, use_pct=True,
               chi_grid=chi_grid)
    plot_panel(main_axes[2], summary, 'F_mean', 'F_ci',
               ylabel=r'Per-firm rewiring count $F$',
               title='(c)', points_to_plot=points_to_plot, chi_grid=chi_grid)

    if not args.main_only:
        # (d) termination breakdown: frac_converged (solid, errorbar) +
        # frac_cycled (dashed, smaller markers).
        plot_panel(diag_axes[0], summary,
                   'frac_conv_mean', 'frac_conv_ci',
                   ylabel='Termination fraction',
                   title='(d)', points_to_plot=points_to_plot,
                   chi_grid=chi_grid,
                   extra=[('frac_cyc_mean', 'frac_cyc_ci', 'cycled', '--')])
        diag_axes[0].set_ylim(-0.03, 1.03)

        # (e) matched-pair retention vs chi=0.
        if mp_summary.empty:
            diag_axes[1].set_axis_off()
            diag_axes[1].text(0.5, 0.5, '(no chi=0 baseline trials)',
                              transform=diag_axes[1].transAxes,
                              ha='center', va='center', color='gray')
        else:
            plot_panel(diag_axes[1], mp_summary, 'M_mean', 'M_ci',
                       ylabel=r'Matched-pair retention vs $\chi=0$',
                       title='(e)', points_to_plot=points_to_plot,
                       chi_grid=chi_grid)
            diag_axes[1].set_ylim(-0.03, 1.03)

        # (f) sample-size info panel.
        render_info_panel(diag_axes[2], summary, mp_summary, points_to_plot)

    if args.log_x:
        flat_axes = (list(main_axes) +
                     (list(diag_axes[:2]) if not args.main_only else []))
        for ax in flat_axes:
            ax.set_xscale('symlog', linthresh=5e-4)

    # Shared legend below the panels.
    handles, labels = main_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=max(1, len(handles)),
               bbox_to_anchor=(0.5, -0.04), frameon=False, fontsize=10)

    fig.savefig(args.output, dpi=200, bbox_inches='tight')
    print(f"\nFigure -> {args.output}")
    pdf_out = os.path.splitext(args.output)[0] + '.pdf'
    fig.savefig(pdf_out, bbox_inches='tight')
    print(f"Figure -> {pdf_out}")


if __name__ == '__main__':
    main()
