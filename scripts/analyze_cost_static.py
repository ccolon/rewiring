"""Analyze the static-gap (option-B) cost-reduction study.

Reads trial-level and firm-level CSVs produced by `cost_reduction_study.py`
*after* the static-gap extension. Expects these extra columns:

  trial-level: unstable_trial, cycled_trial,
               sum_p_current, sum_p_static_best, theta_static, R_window
  firm-level:  p_current_i, p_best_static_i, theta_static_i,
               swaps_first_R, swaps_last_R, swaps_in_cycle

Produces two figures and console diagnostics:

  Figure 1  -- theta_static vs tau, 2 panels (homo/hetero), one curve per
               operating point, parallel to plot_visibility_2panel.py.
  Figure 2  -- per-firm theta_static_i vs tier_i (binned), with OLS, per
               op-point (heterogeneous-tau trials only).

Console
  - per-cell summary (mean theta_static +/- 1.96 SEM, n trials, %unstable,
    %cycled);
  - per-cell per-firm rewire-window breakdown (mean swaps_first_R,
    swaps_last_R, swaps_in_cycle) restricted to cycled trials, plus the
    fraction of swaps in the last cycle period;
  - per-op-point OLS theta_static_i ~ tier_i + degree_out_init_i.

Usage:
    python scripts/analyze_cost_static.py results_cost_static
    python scripts/analyze_cost_static.py results_cost_static --output figs/
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
# IO
# -----------------------------------------------------------------------------

def load_trials(data_dir):
    paths = [p for p in glob.glob(os.path.join(data_dir, '*.csv'))
             if not p.endswith('_firm.csv')]
    if not paths:
        raise FileNotFoundError(f"No trial-level CSVs found in {data_dir}")
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def load_firms(data_dir):
    paths = glob.glob(os.path.join(data_dir, '*_firm.csv'))
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_csv(p) for p in paths]
    return pd.concat(dfs, ignore_index=True)


def _cfg_str(s):
    """Compact 'hom:VAL' or 'unif:LO:HI'."""
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


# Operating-point definitions for the figure (parallel to
# analyze_cost_reduction.COST_GAP_OP_DEFS).
OP_DEFS = [
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


# -----------------------------------------------------------------------------
# Figure 1: theta_static vs tau, 2 panels
# -----------------------------------------------------------------------------

def fig_theta_static_2panel(trials, save_path):
    """Two-panel figure:
        (a) homogeneous tau:   x = tau (= tau_mean, tier_std = 0)
        (b) heterogeneous tau: x = tau_mean (tier_std > 0), with the homo tau=0
            point spliced in (the lognormal collapses to delta_0 at mean=0).
    One curve per operating point.
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
    ymins, ymaxs = [], []

    # Panel (a): homogeneous tau
    for _, label, keys, color, marker, ls in OP_DEFS:
        sub = _select_op(trials, keys)
        sub = sub[(sub['mode'] == 'limited') &
                  (sub['tier_std'].fillna(0) == 0) &
                  sub['theta_static'].notna()]
        if len(sub) == 0:
            continue
        g = (sub.groupby('tier_mean')['theta_static']
                .agg(['mean', 'sem', 'count']).sort_index())
        ys = g['mean'].values
        es = 1.96 * g['sem'].fillna(0).values
        ymins.append(np.nanmin(ys - es)); ymaxs.append(np.nanmax(ys + es))
        axes[0].errorbar(g.index.values, ys, yerr=es,
                         fmt=marker, ls=ls, color=color,
                         markersize=7, lw=1.6, capsize=3,
                         label=label, alpha=0.95)

    # Panel (b): heterogeneous tau, spliced with homo tau=0
    for _, label, keys, color, marker, ls in OP_DEFS:
        sub = _select_op(trials, keys)
        hetero = sub[(sub['mode'] == 'limited') &
                     (sub['tier_std'].fillna(0) > 0)]
        if len(hetero) == 0:
            continue
        tau0 = sub[(sub['mode'] == 'limited') &
                   (sub['tier_std'].fillna(0) == 0) &
                   (sub['tier_mean'] == 0)]
        sub_h = pd.concat([tau0, hetero], ignore_index=True, sort=False)
        sub_h = sub_h[sub_h['theta_static'].notna()]
        if len(sub_h) == 0:
            continue
        g = (sub_h.groupby('tier_mean')['theta_static']
                  .agg(['mean', 'sem', 'count']).sort_index())
        ys = g['mean'].values
        es = 1.96 * g['sem'].fillna(0).values
        ymins.append(np.nanmin(ys - es)); ymaxs.append(np.nanmax(ys + es))
        axes[1].errorbar(g.index.values, ys, yerr=es,
                         fmt=marker, ls=ls, color=color,
                         markersize=7, lw=1.6, capsize=3,
                         label=label, alpha=0.95)

    axes[0].set_title(r'(a) Homogeneous $\tau$', loc='left')
    axes[0].set_xlabel(r'Tier visibility $\tau$')
    axes[0].set_ylabel(r'Static gap $\theta_{\rm static}$')
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    axes[0].set_xticks([0, 1, 2])
    axes[0].grid(alpha=0.3)
    axes[0].axhline(0, color='black', lw=0.6, alpha=0.4)

    axes[1].set_title(r'(b) Heterogeneous $\tau$ (lognormal, std = mean)',
                      loc='left')
    axes[1].set_xlabel(r'Mean tier visibility $\bar{\tau}$')
    axes[1].set_xticks([0, 2, 3])
    axes[1].grid(alpha=0.3)
    axes[1].axhline(0, color='black', lw=0.6, alpha=0.4)
    axes[1].set_ylabel('')
    axes[1].tick_params(labelleft=False)

    if ymins:
        lo, hi = min(ymins), max(ymaxs)
        pad = 0.1 * max(hi - lo, 1e-4)
        axes[0].set_ylim(lo - pad, hi + pad)

    h, l = axes[0].get_legend_handles_labels()
    if h:
        axes[0].legend(h, l, loc='center right', framealpha=0.95)

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Figure 2: per-firm theta_static_i vs tier_i (binned + OLS), per op-point
# -----------------------------------------------------------------------------

def _bin_tau(t, cap=7):
    return min(int(t), cap)


def fig_perfirm_theta_static(firms_h, op_label, save_path):
    """One panel: mean theta_static_i (binned by tier_i) with OLS overlay."""
    if len(firms_h) == 0:
        return
    firms_h = firms_h.copy()
    firms_h['tier_bin'] = firms_h['tier_i'].apply(lambda t: _bin_tau(t, 7))
    g = firms_h.groupby('tier_bin')['theta_static_i']
    bins = sorted(g.groups.keys())
    means = g.mean().reindex(bins)
    sems  = g.sem().reindex(bins)
    counts = g.size().reindex(bins)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.errorbar(bins, means.values, yerr=1.96 * sems.fillna(0).values,
                fmt='o-', lw=1.6, ms=7, capsize=3, color='C2')
    ax.set_xlabel(r'Firm visibility $\tau_i$ (binned; 7 = $\geq 7$)')
    ax.set_ylabel(r'mean per-firm static gap $\theta_{\rm static,i}$')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=3))
    ax.set_title(f'Per-firm static gap vs own visibility  |  {op_label}')
    ax.grid(alpha=0.3); ax.axhline(0, color='black', lw=0.6, alpha=0.3)

    # OLS: theta_static_i ~ const + tier_i + degree_out_init_i
    info = ''
    try:
        X = firms_h[['tier_i', 'degree_out_init_i']].values.astype(float)
        X = np.column_stack([np.ones(len(X)), X])
        y = firms_h['theta_static_i'].values.astype(float)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        sigma2 = (resid ** 2).sum() / max(len(y) - X.shape[1], 1)
        cov = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(cov))
        info = (f"OLS: theta_static_i = {beta[0]:.4g} + {beta[1]:.4g}·tau_i "
                f"+ {beta[2]:.4g}·deg_out_init_i  "
                f"(SE: {se[1]:.4g}, {se[2]:.4g}; N={len(y)})")
        ax.text(0.02, 0.98, info, transform=ax.transAxes, fontsize=7,
                va='top', ha='left',
                bbox=dict(facecolor='white', edgecolor='lightgray', alpha=0.9))
        print(f"  {op_label}: {info}")
    except Exception as e:
        print(f"  {op_label}: OLS failed: {e}")

    # n annotation
    for tau in bins:
        ax.annotate(f'n={int(counts.loc[tau])}', xy=(tau, ax.get_ylim()[0]),
                    xytext=(0, -28), textcoords='offset points',
                    ha='center', fontsize=7, color='gray')

    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Console diagnostics
# -----------------------------------------------------------------------------

def print_per_cell_summary(trials):
    print("\n=== Per-cell summary (theta_static, stability) ===")
    cols = ['n', 'aisi_spread', 'b_short', 'mode', 'tier_mean', 'tier_std']
    for keys, sub in trials.groupby(cols):
        n_, aisi, bcfg, mode, tm, ts = keys
        nt = len(sub)
        ts_mean = sub['theta_static'].mean()
        ts_sem  = sub['theta_static'].sem()
        unst    = 100.0 * sub['unstable_trial'].mean()
        cyc     = 100.0 * sub['cycled_trial'].mean()
        conv    = 100.0 * sub['converged'].mean()
        rounds  = sub['rounds'].mean()
        print(f"  n={n_:>3} aisi={aisi:.3f} b={bcfg:14} mode={mode:7} "
              f"tau=({tm},{ts})  N={nt:>4}  "
              f"theta_static={ts_mean:.4f}±{1.96*ts_sem:.4f}  "
              f"%conv={conv:5.1f} %cyc={cyc:5.1f} %unst={unst:5.1f} "
              f"rounds={rounds:5.1f}")


def print_cycle_breakdown(trials, firms):
    """Per cell, restricted to cycled trials: mean per-firm swaps in the first-R
    window, the last-R window, and the last cycle-period window."""
    if len(firms) == 0:
        return
    print("\n=== Per-cell rewire-window breakdown (cycled trials only) ===")
    cols = ['n', 'aisi_spread', 'b_short', 'mode', 'tier_mean', 'tier_std']
    for keys, sub_trials in trials.groupby(cols):
        cycled = sub_trials[sub_trials['cycled_trial'] == 1]
        if len(cycled) == 0:
            continue
        n_, aisi, bcfg, mode, tm, ts = keys
        # Join cycled trials to firms on (tech_seed, init_seed).
        kfirm = firms.merge(
            cycled[['tech_seed', 'init_seed']].drop_duplicates(),
            on=['tech_seed', 'init_seed']
        )
        if len(kfirm) == 0:
            continue
        sf = kfirm['swaps_first_R'].mean()
        sl = kfirm['swaps_last_R'].mean()
        sc = kfirm['swaps_in_cycle'].mean()
        # Cycle period stats from trials
        cp_med  = cycled['cycle_period'].astype(int).median()
        cp_mean = cycled['cycle_period'].astype(int).mean()
        # Avg total swaps per firm over the whole run
        if 'n_swaps_i' in kfirm.columns:
            ntot = kfirm['n_swaps_i'].mean()
            frac_in_cycle = sc / ntot if ntot > 0 else float('nan')
        else:
            frac_in_cycle = float('nan')
        print(f"  n={n_:>3} aisi={aisi:.3f} b={bcfg:14} mode={mode:7} "
              f"tau=({tm},{ts})  cycled={len(cycled):>3}  "
              f"period(med={cp_med:.0f} mean={cp_mean:.1f})  "
              f"mean swaps/firm  first{int(sub_trials['R_window'].iloc[0])}={sf:.2f}  "
              f"last={sl:.2f}  in-cycle={sc:.2f} "
              f"(frac of total = {frac_in_cycle:.2f})")


def print_unstable_breakdown(trials):
    """Same shape but for unstable (non-converged, non-cycled) trials."""
    print("\n=== Per-cell unstable-trial counts ===")
    cols = ['n', 'aisi_spread', 'b_short', 'mode', 'tier_mean', 'tier_std']
    for keys, sub in trials.groupby(cols):
        unst = sub[sub['unstable_trial'] == 1]
        if len(unst) == 0:
            continue
        n_, aisi, bcfg, mode, tm, ts = keys
        print(f"  n={n_:>3} aisi={aisi:.3f} b={bcfg:14} mode={mode:7} "
              f"tau=({tm},{ts})  N_unstable={len(unst):>3} / {len(sub):>3} "
              f"  mean rounds={unst['rounds'].mean():.1f}  "
              f"mean total_rewirings={unst['total_rewirings'].mean():.1f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir',
                   help='Directory containing coststat_*.csv (and *_firm.csv).')
    p.add_argument('--output', default=None,
                   help='Output figures dir (default: <data_dir>/figures).')
    args = p.parse_args()

    out_dir = args.output or os.path.join(args.data_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # Unicode-safe console on Windows
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    trials = load_trials(args.data_dir)
    firms  = load_firms(args.data_dir)
    print(f"Loaded {len(trials)} trial rows, {len(firms)} firm rows "
          f"from {args.data_dir}")

    # Defensive: the static-gap columns should exist.
    required_trial = ['theta_static', 'unstable_trial', 'cycled_trial', 'R_window']
    missing = [c for c in required_trial if c not in trials.columns]
    if missing:
        print(f"WARNING: missing trial-level columns {missing}. The CSVs may "
              f"come from an older version of cost_reduction_study.py.")

    trials['b_short'] = trials['b_config'].apply(_cfg_str)

    # ---- Figure 1 -----------------------------------------------------------
    fig_theta_static_2panel(
        trials, save_path=os.path.join(out_dir, 'fig_theta_static_2panel.png'))

    # ---- Figure 2 (per op-point, hetero trials only) ------------------------
    if len(firms) > 0:
        # Need b_short on the trial-merge keys.
        for op_id, op_label_str, keys, *_ in OP_DEFS:
            sub_trials = _select_op(trials, keys)
            hetero_keys = sub_trials[(sub_trials['mode'] == 'limited') &
                                     (sub_trials['tier_std'].fillna(0) > 0)] \
                          [['tech_seed', 'init_seed']].drop_duplicates()
            if len(hetero_keys) == 0:
                continue
            firms_h = firms.merge(hetero_keys, on=['tech_seed', 'init_seed'])
            if len(firms_h) == 0 or 'theta_static_i' not in firms_h.columns:
                continue
            firms_h = firms_h[firms_h['theta_static_i'].notna()]
            fig_perfirm_theta_static(
                firms_h, op_label_str,
                os.path.join(out_dir, f'fig_perfirm_theta_static_{op_id}.png'))

    # ---- Console diagnostics ------------------------------------------------
    print_per_cell_summary(trials)
    print_unstable_breakdown(trials)
    print_cycle_breakdown(trials, firms)


if __name__ == '__main__':
    main()
