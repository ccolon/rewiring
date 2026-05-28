"""Audit script for panel (c) of figure_visibility.png.

Reproduces panel (c) (mean relative per-firm cost gap vs binned tier_i,
three R1/R2/R3 series on the same axes) but restricts the data in two
ways, producing two diagnostic figures alongside this script:

  fig_audit_perfirm_by_cell.png
      Three sub-panels, one per heterogeneous-tau cell:
      (bar_tau, sigma_tau) in { (1,1), (2,2), (3,3) }.
      Each sub-panel overlays R1, R2, R3.

  fig_audit_perfirm_by_trial.png
      Three sub-panels, one per trial classification: converged
      (fixed point), limit cycle (period >= 2), unstable (round
      budget exhausted with neither fixed point nor cycle detected).
      Each sub-panel overlays R1, R2, R3, pooling cells.

Both figures share the y-axis across sub-panels for direct comparison.

The main plot_visibility.py script is NOT modified. This file re-uses
its module-level helpers (SERIES_DEFS, _cfg_str_any, _bin_tau).

Usage (from anywhere):
    python results/visibility/audit_perfirm.py
    python results/visibility/audit_perfirm.py --cost_dir other/dir
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from plot_visibility import SERIES_DEFS, _cfg_str_any, _bin_tau  # noqa: E402


# -----------------------------------------------------------------------------
# Data loading (same as plot_visibility.load_cost but also propagates the
# trial-level convergence flags onto firm rows so we can classify by
# converged / cycle / unstable).
# -----------------------------------------------------------------------------

def load_cost_tagged(cost_dir):
    trial_paths = sorted([p for p in glob.glob(os.path.join(cost_dir, '*.csv'))
                          if not p.endswith('_firm.csv')])
    if not trial_paths:
        raise FileNotFoundError(f"No trial CSVs in {cost_dir!r}")
    context_cols_wanted = [
        'tier_mean', 'tier_std', 'mode', 'n', 'cc',
        'max_swaps', 'aisi_spread', 'sigma_w',
        'a_config', 'b_config', 'z_config',
        'converged', 'cycle_period',
    ]
    all_firms, n_pairs = [], 0
    for tp in trial_paths:
        firm_path = tp[:-4] + '_firm.csv'
        if not os.path.exists(firm_path):
            continue
        t = pd.read_csv(tp)
        f = pd.read_csv(firm_path)
        ctx = [c for c in context_cols_wanted
               if c in t.columns and c not in f.columns]
        f_tagged = f.merge(
            t[['tech_seed', 'init_seed'] + ctx],
            on=['tech_seed', 'init_seed'], how='left',
        )
        all_firms.append(f_tagged)
        n_pairs += 1
    firms = pd.concat(all_firms, ignore_index=True)
    # a_short / b_short / z_short for SERIES_DEFS-style filtering.
    for col_short, col_cfg in [('a_short', 'a_config'),
                                ('b_short', 'b_config'),
                                ('z_short', 'z_config')]:
        if col_cfg in firms.columns:
            firms[col_short] = firms[col_cfg].apply(_cfg_str_any)

    # Derive trial_type from converged + cycle_period (robust to schema age).
    cp = pd.to_numeric(firms.get('cycle_period'), errors='coerce')
    conv_col = pd.to_numeric(firms.get('converged'), errors='coerce').fillna(0)
    is_conv = (conv_col == 1) | (cp == 1)
    is_cyc = (~is_conv) & (cp >= 2)
    firms['trial_type'] = np.where(is_conv, 'converged',
                          np.where(is_cyc, 'cycle', 'unstable'))

    print(f"Loaded {n_pairs} trial+firm CSV pairs from {cost_dir}")
    return firms


# -----------------------------------------------------------------------------
# Shared sub-panel plotter (same logic as plot_visibility.plot_perfirm_cost_panel,
# but takes a pre-filtered firm slice and skips the legend).
# -----------------------------------------------------------------------------

def _filter_series(firms, keys):
    """Filter firms to a single visibility-study series (R1/R2/R3)."""
    return firms[
        (firms['n'] == keys['n']) &
        (firms['cc'] == keys['cc']) &
        (firms['a_short'] == keys['a']) &
        (firms['b_short'] == keys['b']) &
        (firms['z_short'] == keys['z']) &
        np.isclose(firms['aisi_spread'].fillna(0), keys['aisi'], atol=1e-9) &
        np.isclose(firms['sigma_w'].fillna(0),    keys['sw'],   atol=1e-9) &
        (firms['mode'] == 'limited')
    ].copy()


def _plot_perseries_subpanel(ax, firms_subset, title, series_ids=('R1', 'R2', 'R3')):
    """Plot one sub-panel: three series overlaid on the pre-filtered slice."""
    plotted_bins = set()
    plotted = False
    counts_per_series = []
    for sid, label, keys, color in SERIES_DEFS:
        if sid not in series_ids:
            continue
        sub = _filter_series(firms_subset, keys)
        sub = sub.dropna(subset=['p_current_i', 'p_best_static_i'])
        counts_per_series.append((sid, len(sub)))
        if len(sub) == 0:
            continue
        sub['theta_rel_i'] = ((sub['p_current_i'] - sub['p_best_static_i'])
                              / sub['p_current_i'])
        sub['tier_bin'] = sub['tier_i'].apply(lambda t: _bin_tau(t, 7))
        g = sub.groupby('tier_bin')['theta_rel_i']
        bins = sorted(g.groups.keys())
        means = g.mean().reindex(bins)
        sems = g.sem().reindex(bins)
        ax.errorbar(bins, means.values, yerr=1.96 * sems.fillna(0).values,
                    fmt='o-', lw=1.6, ms=6, capsize=3, color=color,
                    alpha=0.92, label=label)
        plotted_bins.update(bins)
        plotted = True

    ax.set_title(title, loc='left')
    if not plotted:
        ax.text(0.5, 0.5, '(no data)', transform=ax.transAxes,
                ha='center', va='center', color='gray')
        return counts_per_series

    ax.set_xlabel(r'Firm visibility $\tau_i$')
    ax.set_ylabel(r'Mean relative per-firm cost gap')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    bs = sorted(plotted_bins)
    ax.set_xticks(bs)
    ax.set_xticklabels([(r'$\geq 7$' if b == 7 else str(b)) for b in bs])
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', lw=0.6, alpha=0.3)
    return counts_per_series


# -----------------------------------------------------------------------------
# Audit figures
# -----------------------------------------------------------------------------

def make_audit_by_cell(firms, save_path):
    cells = [(1, 1), (2, 2), (3, 3)]
    fig, axes = plt.subplots(1, len(cells), figsize=(14.5, 4.4),
                             constrained_layout=True, sharey=True)
    print()
    print("== Audit by hetero cell ==")
    for i, (m, s) in enumerate(cells):
        cell_mask = (
            np.isclose(firms['tier_mean'].fillna(-1), m, atol=1e-9) &
            np.isclose(firms['tier_std'].fillna(-1),  s, atol=1e-9)
        )
        sub = firms[cell_mask]
        title = (rf'({chr(97 + i)}) $(\bar\tau, \sigma_\tau)=({m},{s})$, '
                 rf'N={len(sub)} firm-rows')
        per_series_counts = _plot_perseries_subpanel(axes[i], sub, title)
        print(f"  cell (m={m}, s={s}): {len(sub)} firm-rows total")
        for sid, n in per_series_counts:
            print(f"      {sid}: {n} firm-rows")
    for ax in axes[1:]:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)
    h, l = axes[-1].get_legend_handles_labels()
    if h:
        axes[-1].legend(h, l, loc='upper right', framealpha=0.95)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


def make_audit_by_trial(firms, save_path):
    trial_order = [('converged', 'Converged trajectories'),
                   ('cycle',     'Limit-cycle trajectories'),
                   ('unstable',  'Unstable trajectories')]
    fig, axes = plt.subplots(1, len(trial_order), figsize=(14.5, 4.4),
                             constrained_layout=True, sharey=True)
    print()
    print("== Audit by trial type ==")
    for i, (tt, pretty) in enumerate(trial_order):
        sub = firms[firms['trial_type'] == tt]
        title = (rf'({chr(97 + i)}) {pretty}, '
                 rf'N={len(sub)} firm-rows')
        per_series_counts = _plot_perseries_subpanel(axes[i], sub, title)
        print(f"  trial_type = {tt}: {len(sub)} firm-rows total")
        for sid, n in per_series_counts:
            print(f"      {sid}: {n} firm-rows")
    for ax in axes[1:]:
        ax.set_ylabel('')
        ax.tick_params(labelleft=False)
    h, l = axes[-1].get_legend_handles_labels()
    if h:
        axes[-1].legend(h, l, loc='upper right', framealpha=0.95)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {save_path}")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cost_dir', default=None,
                   help='Cost-gap CSVs dir '
                        '(default: <script_dir>/../cost_gap).')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    cost_dir = (args.cost_dir or
                os.path.normpath(os.path.join(SCRIPT_DIR, os.pardir, 'cost_gap')))
    firms = load_cost_tagged(cost_dir)

    print()
    print(f"Total firm-rows: {len(firms)}")
    print("Per (tier_mean, tier_std) cell counts:")
    print(firms.groupby(['tier_mean', 'tier_std']).size()
                .to_string())
    print()
    print("Per trial_type counts:")
    print(firms.groupby('trial_type').size().to_string())
    print()
    print("Cross-tab cell vs trial_type:")
    print(firms.groupby(['tier_mean', 'tier_std', 'trial_type'])
                .size().unstack(fill_value=0).to_string())

    make_audit_by_cell(
        firms, os.path.join(SCRIPT_DIR, 'fig_audit_perfirm_by_cell.png'))
    make_audit_by_trial(
        firms, os.path.join(SCRIPT_DIR, 'fig_audit_perfirm_by_trial.png'))


if __name__ == '__main__':
    main()
