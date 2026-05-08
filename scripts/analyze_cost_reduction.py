"""Analyze the cost-reduction (theta_T vs tau) study.

Reads trial-level and firm-level CSVs from `cost_reduction_study.py` and
produces three figures:

  Figure 1 (main):   theta_T vs tau (homogeneous tau scan + full-mode reference).
  Figure 2 (appendix, per-firm): mean cost-gap and rewire count vs tier_i,
                                  binned, from heterogeneous-tau cells.
                                  Plus a regression cost_gap_i ~ tier_i + degree_out.
  Figure 3 (robustness): theta_T vs tau overlaid across operating points.

Usage:
    python scripts/analyze_cost_reduction.py results/cost
    python scripts/analyze_cost_reduction.py results_cost --output figures_cost
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

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_trial_csvs(data_dir):
    """Load and concatenate all trial-level CSVs (excluding *_firm.csv)."""
    paths = [p for p in glob.glob(os.path.join(data_dir, '*.csv'))
             if not p.endswith('_firm.csv')]
    if not paths:
        raise FileNotFoundError(f"No trial-level CSVs found in {data_dir}")
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def load_firm_csvs(data_dir):
    """Load and concatenate all *_firm.csv files."""
    paths = glob.glob(os.path.join(data_dir, '*_firm.csv'))
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def op_point_label(row):
    """Compact label for an operating-point row (one of: A, B, C, ...)."""
    if 'homogeneous:0.9' in str(row['b_config']) and row['aisi_spread'] == 0.0 and row['n'] == 50:
        return 'opA: n=50, b=0.9 hom, no AiSi'
    if 'uniform:0.9:1.1' in str(row['b_config']) and row['aisi_spread'] > 0:
        return f"opB: n={row['n']}, uniform a/b/z, aisi={row['aisi_spread']}"
    if 'homogeneous:0.9' in str(row['b_config']) and row['n'] == 100:
        return 'opC: n=100, b=0.9 hom, no AiSi'
    return f"n={row['n']}, b={row['b_config']}, aisi={row['aisi_spread']}"


def _cfg_str(s):
    """Compact 'hom:VAL' or 'unif:LO:HI' from either a JSON config dict
    (diversity_study.py output) or the CLI form 'homogeneous:V' /
    'uniform:LO:HI' (cost_reduction_study.py output)."""
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
    parts = s.split(':')
    if parts[0] in ('homogeneous', 'hom') and len(parts) >= 2:
        return f"hom:{float(parts[1])}"
    if parts[0] in ('uniform', 'unif') and len(parts) >= 3:
        return f"unif:{float(parts[1])}:{float(parts[2])}"
    return 'NA'


# -----------------------------------------------------------------------------
# Option A: cost-gap metric using the matched-init full-mode reference
# -----------------------------------------------------------------------------

def compute_cost_gap_a(trials):
    """Add `cost_gap_A` column to `trials`, defined as

        cost_gap_A = (sum_p_final_window_limited - sum_p_final_window_full)
                     / sum_p_final_window_limited

    where the full-mode reference is matched on the full set of
    cell-defining parameters AND (tech_seed, init_seed). Rows without a
    matching full-mode reference get NaN.

    Important: matching on (tech_seed, init_seed) alone is unsafe because
    cost_reduction_study.py reuses the same tech_seed across n values, so
    n=50 and n=100 share tech_seed numbers but represent different economies.
    The full key resolves this.
    """
    trials = trials.copy()
    full = trials[trials['mode'] == 'full']
    if len(full) == 0:
        trials['sum_p_full_match'] = np.nan
        trials['cost_gap_A']       = np.nan
        return trials

    key_cols = ['n', 'cc', 'aisi_spread', 'sigma_w',
                'a_config', 'b_config', 'z_config',
                'max_swaps', 'tech_seed', 'init_seed']
    full_lookup = (full.drop_duplicates(key_cols)
                       [key_cols + ['sum_p_final_window']]
                       .rename(columns={'sum_p_final_window': 'sum_p_full_match'}))

    trials = trials.merge(full_lookup, on=key_cols, how='left')
    trials['cost_gap_A'] = ((trials['sum_p_final_window'] -
                              trials['sum_p_full_match']) /
                             trials['sum_p_final_window'])
    return trials


def print_cost_gap_coverage(trials):
    print("\n=== cost_gap_A coverage (option A: matched-init full-mode reference) ===")
    lim = trials[trials['mode'] == 'limited']
    n_total   = len(lim)
    n_with_full = lim['cost_gap_A'].notna().sum()
    print(f"  limited rows total: {n_total};  with matched full-mode ref: "
          f"{n_with_full}  ({100 * n_with_full / max(n_total, 1):.1f}%)")
    if n_with_full == 0:
        return
    op_label_col = lim.apply(op_point_label, axis=1)
    for op, idxs in op_label_col.groupby(op_label_col).groups.items():
        sub = lim.loc[idxs]
        n = sub['cost_gap_A'].notna().sum()
        if n == 0:
            continue
        print(f"  {op[:55]:55} | {n} usable cost_gap_A rows; "
              f"mean = {sub['cost_gap_A'].mean():.4f}, "
              f"min = {sub['cost_gap_A'].min():.4f}, "
              f"max = {sub['cost_gap_A'].max():.4f}")


# -----------------------------------------------------------------------------
# 2-panel cost-reduction-potential figure (option A)
# -----------------------------------------------------------------------------

# Operating-point definitions for the 2-panel figure.  Each entry:
#   (id, label, key dict, color, marker, linestyle)
COST_GAP_OP_DEFS = [
    ('opA', r'Homogeneous DRS, $n=50$',
     {'n': 50,  'b_short': 'hom:0.9',      'aisi_spread': 0.0},
     'C2', 'o', '-'),
    ('opB', r'Full heterogeneity, $n=50$',
     {'n': 50,  'b_short': 'unif:0.9:1.1', 'aisi_spread': 0.05},
     'C0', 's', '-'),
    ('opC', r'Homogeneous DRS, $n=100$',
     {'n': 100, 'b_short': 'hom:0.9',      'aisi_spread': 0.0},
     'C2', 'D', '--'),
]


def _select_op(trials, keys):
    sub = trials.copy()
    for col, val in keys.items():
        if col == 'aisi_spread':
            sub = sub[np.isclose(sub['aisi_spread'], val, atol=1e-9)]
        else:
            sub = sub[sub[col] == val]
    return sub


def fig_cost_gap_2panel(trials, save_path):
    """Two-panel cost-reduction-potential figure (option A), parallel layout
    to plot_visibility_2panel.py:

        (a) Homogeneous tau:    x = tau_i (= tau for all i)
        (b) Heterogeneous tau:  x = mean tau (lognormal, std = mean)

    Series: one curve per operating point.
    """
    if 'b_short' not in trials.columns:
        trials = trials.copy()
        trials['b_short'] = trials['b_config'].apply(_cfg_str)

    plt.rcParams.update({
        'font.size':        11,
        'axes.titlesize':   12,
        'axes.labelsize':   11,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'legend.fontsize':  9.5,
    })

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.0),
                              constrained_layout=True, sharey=True)
    # Will compute a data-driven ylim after plotting, so leave for now.
    plotted_ymins, plotted_ymaxs = [], []

    # Panel (a): homogeneous tau
    for _, label, keys, color, marker, ls in COST_GAP_OP_DEFS:
        sub = _select_op(trials, keys)
        sub = sub[(sub['mode'] == 'limited') &
                  (sub['tier_std'].fillna(0) == 0) &
                  sub['cost_gap_A'].notna()]
        if len(sub) == 0:
            continue
        g = (sub.groupby('tier_mean')['cost_gap_A']
                  .agg(['mean', 'sem', 'count'])
                  .sort_index())
        ys = g['mean'].values
        es = 1.96 * g['sem'].fillna(0).values
        plotted_ymins.append(np.nanmin(ys - es))
        plotted_ymaxs.append(np.nanmax(ys + es))
        axes[0].errorbar(g.index.values, ys, yerr=es,
                         fmt=marker, ls=ls, color=color,
                         markersize=7, lw=1.6, capsize=3,
                         label=label, alpha=0.95)

    # Panel (b): heterogeneous tau, with homo tau=0 spliced in for the
    # leftmost point (lognormal degenerates to delta_0 when its mean is 0).
    for _, label, keys, color, marker, ls in COST_GAP_OP_DEFS:
        sub = _select_op(trials, keys)
        hetero = sub[(sub['mode'] == 'limited') &
                     (sub['tier_std'].fillna(0) > 0)]
        if len(hetero) == 0:
            continue
        tau0 = sub[(sub['mode'] == 'limited') &
                   (sub['tier_std'].fillna(0) == 0) &
                   (sub['tier_mean'] == 0)]
        sub_h = pd.concat([tau0, hetero], ignore_index=True, sort=False)
        sub_h = sub_h[sub_h['cost_gap_A'].notna()]
        if len(sub_h) == 0:
            continue
        g = (sub_h.groupby('tier_mean')['cost_gap_A']
                    .agg(['mean', 'sem', 'count'])
                    .sort_index())
        ys = g['mean'].values
        es = 1.96 * g['sem'].fillna(0).values
        plotted_ymins.append(np.nanmin(ys - es))
        plotted_ymaxs.append(np.nanmax(ys + es))
        axes[1].errorbar(g.index.values, ys, yerr=es,
                         fmt=marker, ls=ls, color=color,
                         markersize=7, lw=1.6, capsize=3,
                         label=label, alpha=0.95)

    # Per-panel formatting
    axes[0].set_title(r'(a) Homogeneous $\tau$', loc='left')
    axes[0].set_xlabel(r'Tier visibility $\tau$')
    axes[0].set_ylabel('Cost-reduction potential')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    axes[0].set_xticks([0, 1, 2, 3, 4, 5, 6])
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color='black', lw=0.6, alpha=0.4)

    axes[1].set_title(r'(b) Heterogeneous $\tau$ (lognormal, std = mean)',
                      loc='left')
    axes[1].set_xlabel(r'Mean tier visibility $\bar{\tau}$')
    axes[1].set_xticks([0, 1, 2, 3, 4, 5, 6])
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color='black', lw=0.6, alpha=0.4)
    axes[1].set_ylabel('')
    axes[1].tick_params(labelleft=False)

    # Data-driven y-range with small padding (sharey, so set once).
    if plotted_ymins:
        lo, hi = min(plotted_ymins), max(plotted_ymaxs)
        pad = 0.1 * max(hi - lo, 1e-4)
        axes[0].set_ylim(lo - pad, hi + pad)

    h, l = axes[0].get_legend_handles_labels()
    if h:
        axes[0].legend(h, l, loc='center right', framealpha=0.95)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 1: theta_T vs tau, main op-point
# -----------------------------------------------------------------------------

def fig1_theta_T_vs_tau(trials, op_label, save_path):
    """One panel: theta_T (sum & util) vs tau, with full-mode horizontal line."""
    # Filter homogeneous-tau rows for this op-point
    homo = trials[(trials['mode'] == 'limited') & (trials['tier_std'].fillna(0) == 0)]
    full = trials[trials['mode'] == 'full']

    # Group by tier_mean
    g = homo.groupby('tier_mean')
    tau_axis = sorted(g.groups.keys())
    theta_sum_mean = g['theta_T_sum'].mean().reindex(tau_axis)
    theta_sum_se   = g['theta_T_sum'].sem().reindex(tau_axis)
    theta_util_mean = g['theta_T_util'].mean().reindex(tau_axis)
    theta_util_se   = g['theta_T_util'].sem().reindex(tau_axis)

    full_sum  = full['theta_T_sum'].mean()  if len(full) else np.nan
    full_util = full['theta_T_util'].mean() if len(full) else np.nan

    fig, ax1 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)

    # Primary axis: theta_T_sum
    ax1.errorbar(tau_axis, theta_sum_mean, yerr=1.96 * theta_sum_se,
                 fmt='o-', color='C0', lw=1.6, ms=7, capsize=3,
                 label=r'$\theta_T^\Sigma = (\Sigma p_{\rm init} - \langle\Sigma p\rangle_{\rm final}) / \Sigma p_{\rm init}$')
    if np.isfinite(full_sum):
        ax1.axhline(full_sum, ls='--', color='C0', alpha=0.5,
                    label=fr'$\theta_T^\Sigma$ at $\tau=\infty$ (full): {full_sum:.3f}')
    ax1.set_xlabel(r'tier visibility $\tau$')
    ax1.set_ylabel(r'$\theta_T^\Sigma$ (aggregate price drop)')
    ax1.grid(alpha=0.3)
    ax1.legend(loc='lower right', fontsize=9)

    # Secondary axis: theta_T_util
    ax2 = ax1.twinx()
    ax2.errorbar(tau_axis, theta_util_mean, yerr=1.96 * theta_util_se,
                 fmt='s--', color='C3', lw=1.4, ms=6, capsize=3, alpha=0.8,
                 label=r'$\theta_T^U = U_{\rm final} - U_{\rm init}$')
    if np.isfinite(full_util):
        ax2.axhline(full_util, ls=':', color='C3', alpha=0.5)
    ax2.set_ylabel(r'$\theta_T^U$ (log price drop, geometric mean)', color='C3')
    ax2.tick_params(axis='y', labelcolor='C3')

    ax1.set_title(f'Cost-reduction potential vs visibility tier  |  {op_label}')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2: per-firm panel from heterogeneous tau cells
# -----------------------------------------------------------------------------

def fig2_per_firm_hetero(trials, firms, op_label, save_path):
    """Two-panel: cost gap and rewire count vs firm tier_i (binned)."""
    # Identify hetero trials in this op-point (tier_std > 0)
    hetero_keys = trials[(trials['mode'] == 'limited') & (trials['tier_std'].fillna(0) > 0)] \
        [['tech_seed', 'init_seed']].drop_duplicates()
    if len(hetero_keys) == 0:
        print(f"  (no hetero-tau trials at op-point '{op_label}', skipping fig2)")
        return

    hetero_firms = firms.merge(hetero_keys, on=['tech_seed', 'init_seed'])
    if len(hetero_firms) == 0:
        return
    hetero_firms = hetero_firms.copy()
    hetero_firms['cost_gap'] = (hetero_firms['p_init_i'] - hetero_firms['p_final_i']) \
                               / hetero_firms['p_init_i']

    # Bin tier_i: 0, 1, 2, ..., 6, 7+
    def bin_tau(t):
        return min(int(t), 7)
    hetero_firms['tier_bin'] = hetero_firms['tier_i'].apply(bin_tau)

    g = hetero_firms.groupby('tier_bin')
    bins = sorted(g.groups.keys())
    cost_mean = g['cost_gap'].mean().reindex(bins)
    cost_se   = g['cost_gap'].sem().reindex(bins)
    rewire_mean = g['n_swaps_i'].mean().reindex(bins)
    rewire_se   = g['n_swaps_i'].sem().reindex(bins)
    counts = g.size().reindex(bins)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axA = axes[0]
    axA.errorbar(bins, cost_mean, yerr=1.96 * cost_se, fmt='o-',
                 lw=1.6, ms=7, capsize=3, color='C2')
    axA.set_xlabel(r'firm visibility $\tau_i$ (binned; 7 = $\geq 7$)')
    axA.set_ylabel(r'mean per-firm cost gap $(p_i^{\rm init} - p_i^{\rm final}) / p_i^{\rm init}$')
    axA.set_title('Per-firm cost gap vs own visibility')
    axA.grid(alpha=0.3)
    axA.axhline(0, color='black', lw=0.8, alpha=0.3)

    axB = axes[1]
    axB.errorbar(bins, rewire_mean, yerr=1.96 * rewire_se, fmt='s-',
                 lw=1.6, ms=7, capsize=3, color='C4')
    axB.set_xlabel(r'firm visibility $\tau_i$ (binned; 7 = $\geq 7$)')
    axB.set_ylabel(r'mean per-firm rewire count')
    axB.set_title('Per-firm rewire count vs own visibility')
    axB.grid(alpha=0.3)

    # Sample-size annotation under the x-axis
    for ax in (axA, axB):
        for tau in bins:
            ax.annotate(f'n={counts.loc[tau]}', xy=(tau, ax.get_ylim()[0]),
                        xytext=(0, -28), textcoords='offset points',
                        ha='center', fontsize=7, color='gray')

    # Regression: cost_gap_i ~ tier_i + degree_out_init_i
    try:
        X = hetero_firms[['tier_i', 'degree_out_init_i']].values.astype(float)
        # Add intercept
        X = np.column_stack([np.ones(len(X)), X])
        y = hetero_firms['cost_gap'].values.astype(float)
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        # Standard errors
        resid = y - X @ beta
        sigma2 = (resid ** 2).sum() / (len(y) - X.shape[1])
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        line = (f"OLS: cost_gap = {beta[0]:.4f} + {beta[1]:.5f}·tier_i "
                f"+ {beta[2]:.5f}·degree_out_init_i  "
                f"(SE: {se[1]:.5f}, {se[2]:.5f})  N={len(y)}")
        axA.text(0.02, 0.98, line, transform=axA.transAxes, fontsize=7,
                 va='top', ha='left',
                 bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.9))
        print(f"  {line}")
    except Exception as e:
        print(f"  regression failed: {e}")

    fig.suptitle(f'Per-firm outcomes by individual visibility (heterogeneous $\\tau$)  |  {op_label}')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 3: theta_T vs tau, op-points overlaid (robustness)
# -----------------------------------------------------------------------------

def fig3_op_point_overlay(trials, save_path):
    """One panel: theta_T_sum vs tau, separate line per op-point."""
    # Categorise op-points by (n, b_config, aisi_spread)
    trials = trials.copy()
    trials['op_label'] = trials.apply(op_point_label, axis=1)
    homo = trials[(trials['mode'] == 'limited') & (trials['tier_std'].fillna(0) == 0)]

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for i, (op, sub) in enumerate(homo.groupby('op_label')):
        g = sub.groupby('tier_mean')['theta_T_sum'].agg(['mean', 'sem', 'count'])
        g = g.sort_index()
        ax.errorbar(g.index, g['mean'], yerr=1.96 * g['sem'], fmt='o-',
                    color=colors[i % len(colors)], lw=1.6, ms=6, capsize=3,
                    label=op)
    ax.set_xlabel(r'tier visibility $\tau$ (homogeneous)')
    ax.set_ylabel(r'$\theta_T^\Sigma$ (aggregate price drop)')
    ax.set_title('Cost-reduction potential vs visibility, across operating points')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Per-cell summary print
# -----------------------------------------------------------------------------

def print_summary(trials):
    print("\n=== Per-cell summary (mean ± 1.96·SEM) ===")
    cols = ['n', 'aisi_spread', 'b_config', 'mode', 'tier_mean', 'tier_std']
    for keys, sub in trials.groupby(cols):
        n_, aisi, bcfg, mode, tm, ts = keys
        n_trials = len(sub)
        theta_sum = sub['theta_T_sum'].mean()
        theta_sum_se = sub['theta_T_sum'].sem()
        theta_util = sub['theta_T_util'].mean()
        rounds = sub['rounds'].mean()
        rew_mean = sub['total_rewirings'].mean()
        conv = sub['converged'].mean()
        print(f"  n={n_:>3}  aisi={aisi:.3f}  b={bcfg[:25]:25}  mode={mode:7}  "
              f"tau=({tm},{ts})  trials={n_trials:>4}  conv={conv:.2f}  "
              f"theta_sum={theta_sum:.4f}±{1.96*theta_sum_se:.4f}  "
              f"theta_util={theta_util:.3f}  rounds={rounds:.1f}  rew={rew_mean:.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir', help='Directory containing cost_*.csv files.')
    p.add_argument('--output', default=None,
                   help='Output figures dir (default: <data_dir>/figures).')
    args = p.parse_args()

    out_dir = args.output or os.path.join(args.data_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Allow non-ASCII in console output on Windows.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    trials = load_trial_csvs(args.data_dir)
    firms  = load_firm_csvs(args.data_dir)
    print(f"Loaded {len(trials)} trials, {len(firms)} firm-level rows")

    print_summary(trials)

    # Compute option-A cost gap (matched-init full-mode reference) and produce
    # the manuscript-faithful 2-panel figure (parallel layout to
    # plot_visibility_2panel.py).
    trials = compute_cost_gap_a(trials)
    print_cost_gap_coverage(trials)
    fig_cost_gap_2panel(trials,
                        save_path=os.path.join(out_dir,
                                               'fig_cost_gap_A_2panel.png'))

    # Categorise op-points
    trials['op_label'] = trials.apply(op_point_label, axis=1)
    op_labels = trials['op_label'].unique()

    # Per op-point fig 1 & fig 2
    for op in op_labels:
        sub_trials = trials[trials['op_label'] == op]
        sub_firms = firms.merge(
            sub_trials[['tech_seed', 'init_seed']].drop_duplicates(),
            on=['tech_seed', 'init_seed']
        ) if len(firms) else firms

        op_safe = op.replace(':', '').replace(',', '_').replace(' ', '_').replace('/', '_')
        fig1_theta_T_vs_tau(sub_trials, op,
                             os.path.join(out_dir, f'fig1_theta_T_{op_safe}.png'))
        if len(sub_firms):
            fig2_per_firm_hetero(sub_trials, sub_firms, op,
                                  os.path.join(out_dir, f'fig2_perfirm_{op_safe}.png'))

    # Cross-op figure
    fig3_op_point_overlay(trials, os.path.join(out_dir, 'fig3_op_overlay.png'))


if __name__ == '__main__':
    main()
