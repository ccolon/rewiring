"""Three-panel visibility figure for the manuscript.

Layout (1 row, 3 columns):
    +---------------------+---------------------+---------------------+
    | (a) homo  tau       | (b) hetero tau      | (c) per-firm        |
    | conv + cycled frac  | conv + cycled frac  | relative cost gap   |
    | R1, R2, R3          | R1, R2, R3          | opC, hetero tau     |
    +---------------------+---------------------+---------------------+

Panels (a) and (b) follow plot_visibility_2panel.py: per series, the solid
line is the fraction of trials reaching a fixed point and the dashed line
the fraction trapped in a limit cycle. Confidence bands are 95% normal
approximations.

Panel (c) is the per-firm view from the cost-gap study. For each firm i
at the trial's final state, we recompute the full GE under every
ms-reachable single-firm swap (other firms held fixed) and record the
minimum-cost candidate p_best_static_i. The plotted quantity is the
RELATIVE gap

    theta_rel_i = (p_current_i - p_best_static_i) / p_current_i

binned by the firm's own tier visibility tau_i. The 7+ bin lumps tau_i
>= 7. Restricted to the opC operating point (n=100, hom DRS, no AiSi)
in heterogeneous-tau trials.

Note: cost_reduction_study.py stores theta_static_i = p_current - p_best
(absolute price diff), not a relative gap. This script recomputes the
relative gap on the fly and prints sanity stats to the console so the
small values can be inspected directly.

Data sources:
  - Visibility CSVs:  results/visibility/   (overridable via --data_dir)
  - Cost-gap CSVs:    results/cost_gap/     (overridable via --cost_dir)

Usage (from anywhere):
    python campaigns/visibility/plot.py
    python campaigns/visibility/plot.py --cost_dir path/to/cost_gap
    python campaigns/visibility/plot.py --output figs/foo.png
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


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Tracked code in campaigns/visibility/; CSVs + output figure live in the
# mirroring results/visibility/ (gitignored). Panel (c) reads cost-gap CSVs
# from results/cost_gap/.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_VIS_DIR = os.path.join(REPO_ROOT, 'results', 'visibility')
DEFAULT_COST_DIR = os.path.join(REPO_ROOT, 'results', 'cost_gap')


# =============================================================================
# Visibility data (panels a, b)
# =============================================================================

def _cfg_str_any(s):
    """Compact 'hom:VAL' or 'unif:LO:HI' from either JSON or CLI form."""
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


SERIES_DEFS = [
    ('R1', 'Full heterogeneity',
     dict(a='unif:0.4:0.6', b='unif:0.9:1.1', z='unif:0.9:1.1',
          aisi=0.05, sw=0.05, n=100, cc=4),
     'C0'),
    ('R2', 'Firm-level heterogeneity only',
     dict(a='unif:0.4:0.6', b='unif:0.9:1.1', z='unif:0.9:1.1',
          aisi=0.0, sw=0.0, n=100, cc=4),
     'C1'),
    ('R3', 'Homogenous DRS',
     dict(a='hom:0.5', b='hom:0.9', z='hom:1.0',
          aisi=0.0, sw=0.0, n=100, cc=4),
     'C2'),
]


def load_visibility(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {data_dir!r}")
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if 'tier_mean' not in d.columns or 'tau_mode' not in d.columns:
            continue
        dfs.append(d)
    if not dfs:
        raise RuntimeError(f"No visibility-format CSVs in {data_dir}")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['a_short'] = df['a_config'].apply(_cfg_str_any)
    df['b_short'] = df['b_config'].apply(_cfg_str_any)
    df['z_short'] = df['z_config'].apply(_cfg_str_any)
    # tier_dist column: legacy CSVs lack it; missing entries mean lognormal.
    if 'tier_dist' not in df.columns:
        df['tier_dist'] = 'lognormal'
    else:
        df['tier_dist'] = df['tier_dist'].fillna('lognormal')
    print(f"[visibility] Loaded {len(dfs)} CSVs -> {len(df)} rows from "
          f"{data_dir}")
    return df


def _select_vis(df, keys):
    return df[(df['a_short'] == keys['a']) &
              (df['b_short'] == keys['b']) &
              (df['z_short'] == keys['z']) &
              (df['n'] == keys['n']) &
              (df['cc'] == keys['cc']) &
              (df['aisi_spread'] == keys['aisi']) &
              (df['sigma_w'] == keys['sw'])]


def _regime_fractions(sub):
    if len(sub) == 0:
        return pd.DataFrame(columns=['tier_mean', 'n', 'p_fix', 'p_cyc',
                                      'p_unst', 'ci_fix', 'ci_cyc', 'ci_unst'])
    s = sub.copy()
    cp = pd.to_numeric(s['cycle_period'], errors='coerce')
    s['is_fix']  = (s['converged'].astype(int) == 1) | (cp == 1)
    s['is_cyc']  = (~s['is_fix']) & (cp >= 2)
    s['is_unst'] = (~s['is_fix']) & (~s['is_cyc'])
    g = s.groupby('tier_mean').agg(
        n=('is_fix', 'size'),
        p_fix=('is_fix', 'mean'),
        p_cyc=('is_cyc', 'mean'),
        p_unst=('is_unst', 'mean'),
    ).reset_index()
    for col in ['p_fix', 'p_cyc', 'p_unst']:
        g[f'ci_{col[2:]}'] = (1.96 *
            np.sqrt((g[col] * (1 - g[col])) / g['n'].clip(lower=1)))
    return g.sort_values('tier_mean')


def plot_visibility_panel(ax, df, tau_mode, title, x_label, series_ids,
                          tier_dist='poisson'):
    """tier_dist: restrict hetero rows to this distribution; homo rows are
    accepted regardless (homo dynamics are identical across tier_dists)."""
    plotted = False
    for sid, label, keys, color in SERIES_DEFS:
        if sid not in series_ids:
            continue
        sub = _select_vis(df, keys)
        if tau_mode == 'homo':
            sub = sub[(sub['tau_mode'] == 'homo') & (sub['tier_std'] == 0)]
        else:
            hetero = sub[(sub['tau_mode'] == 'hetero') &
                         (sub['tier_std'] > 0) &
                         (sub['tier_dist'] == tier_dist)]
            if len(hetero) == 0:
                continue
            # Splice the homogeneous tau=0 endpoint (independent of tier_dist).
            tau0 = sub[(sub['tau_mode'] == 'homo') & (sub['tier_std'] == 0) &
                       (sub['tier_mean'] == 0)]
            sub = pd.concat([tau0, hetero], ignore_index=True, sort=False)
        if len(sub) == 0:
            continue
        g = _regime_fractions(sub)
        x = g['tier_mean'].values
        ax.plot(x, g['p_fix'], 'o-', color=color, lw=1.8, ms=7,
                label=f'{label}  -- converged')
        ax.fill_between(x, g['p_fix'] - g['ci_fix'], g['p_fix'] + g['ci_fix'],
                        color=color, alpha=0.15, linewidth=0)
        ax.plot(x, g['p_cyc'], 'x--', color=color, lw=1.4, ms=7, alpha=0.85,
                label=f'{label}  -- periodic')
        ax.fill_between(x, g['p_cyc'] - g['ci_cyc'], g['p_cyc'] + g['ci_cyc'],
                        color=color, alpha=0.10, linewidth=0)
        plotted = True

    ax.set_xlabel(x_label)
    ax.set_title(title, loc='left')
    ax.set_ylim(-0.03, 1.03)
    ax.set_xticks([0, 1, 2, 3, 4, 5, 6])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(alpha=0.3)
    if not plotted:
        ax.text(0.5, 0.5, '(no data)', transform=ax.transAxes,
                ha='center', va='center', color='gray', fontsize=12)


# =============================================================================
# Cost-gap data (panel c) -- panel keyed off SERIES_DEFS so panels a/b/c
# stay in sync.
# =============================================================================

def load_cost(cost_dir):
    """Load trial+firm CSV pairs and tag every firm row with its cell context.

    Why this matters: the SLURM launcher uses one BASE_SEED for every cell, so
    `(tech_seed, init_seed)` pairs collide across cells. A naive merge on
    those keys would pool firms from all cells.  Reading the trial+firm
    pair file-by-file is unambiguous: within a single CSV pair, each
    `(tech_seed, init_seed)` belongs to one cell.
    """
    trial_paths = sorted([p for p in glob.glob(os.path.join(cost_dir, '*.csv'))
                          if not p.endswith('_firm.csv')])
    if not trial_paths:
        raise FileNotFoundError(f"No trial CSVs in {cost_dir!r}")
    context_cols_wanted = ['tier_mean', 'tier_std', 'tier_dist', 'mode', 'n', 'cc',
                           'max_swaps', 'aisi_spread', 'sigma_w',
                           'a_config', 'b_config', 'z_config']
    all_trials, all_firms, n_pairs = [], [], 0
    for tp in trial_paths:
        firm_path = tp[:-4] + '_firm.csv'
        if not os.path.exists(firm_path):
            continue
        t = pd.read_csv(tp)
        f = pd.read_csv(firm_path)
        context_cols = [c for c in context_cols_wanted
                        if c in t.columns and c not in f.columns]
        # Within-file merge is safe: (tech_seed, init_seed) is unique per row.
        f_tagged = f.merge(t[['tech_seed', 'init_seed'] + context_cols],
                           on=['tech_seed', 'init_seed'], how='left')
        all_trials.append(t)
        all_firms.append(f_tagged)
        n_pairs += 1
    trials = pd.concat(all_trials, ignore_index=True)
    firms = (pd.concat(all_firms, ignore_index=True)
             if all_firms else pd.DataFrame())
    # Add a_short / b_short / z_short on both frames so we can filter by
    # the (a, b, z) keys used in SERIES_DEFS.
    for col_short, col_cfg in [('a_short', 'a_config'),
                                ('b_short', 'b_config'),
                                ('z_short', 'z_config')]:
        if col_cfg in trials.columns:
            trials[col_short] = trials[col_cfg].apply(_cfg_str_any)
        if col_cfg in firms.columns:
            firms[col_short] = firms[col_cfg].apply(_cfg_str_any)
    # tier_dist: legacy CSVs lack it; missing entries default to 'lognormal'.
    for fr in (trials, firms):
        if 'tier_dist' not in fr.columns:
            fr['tier_dist'] = 'lognormal'
        else:
            fr['tier_dist'] = fr['tier_dist'].fillna('lognormal')
    print(f"[cost]       Loaded {n_pairs} trial+firm CSV pairs from {cost_dir}")
    return trials, firms


def _bin_tau(t, cap=7):
    return min(int(t), cap)


def plot_perfirm_cost_panel(ax, firms, series_ids=('R1', 'R2', 'R3')):
    """Right panel: mean RELATIVE per-firm static gap vs binned tier_i,
    one curve per visibility series (R1/R2/R3) -- same series and colours
    as panels (a, b).

    theta_rel_i := (p_current_i - p_best_static_i) / p_current_i, where
    p_best_static_i is the min over candidate supplier sets reachable by
    up to static_max_swaps simultaneous swaps (default = min(c, cc), i.e.
    full enumeration of size-c subsets of the firm's pool, evaluated as
    a full-GE counterfactual with the rest of the network held at the
    trial's final state).

    Per series, included cells are all hetero-tau cells (tier_std > 0)
    PLUS the (tier_mean=0, tier_std=0) endpoint -- the splice that anchors
    the leftmost bar_tau=0 point of panel (b).
    """
    if len(firms) == 0 or 'b_short' not in firms.columns \
            or 'a_short' not in firms.columns:
        ax.text(0.5, 0.5, '(no per-firm data with cell context)',
                transform=ax.transAxes, ha='center', va='center', color='gray')
        return

    print()
    print("[cost] Per-firm relative static gap, by visibility series.")

    plotted_bins = set()
    plotted = False
    for sid, label, keys, color in SERIES_DEFS:
        if sid not in series_ids:
            continue
        sub = firms[
            (firms['n'] == keys['n']) &
            (firms['cc'] == keys['cc']) &
            (firms['a_short'] == keys['a']) &
            (firms['b_short'] == keys['b']) &
            (firms['z_short'] == keys['z']) &
            np.isclose(firms['aisi_spread'].fillna(0), keys['aisi'], atol=1e-9) &
            np.isclose(firms['sigma_w'].fillna(0),    keys['sw'],   atol=1e-9) &
            (firms['mode'] == 'limited') &
            (firms['tier_dist'] == 'poisson')
        ].copy()
        # Include hetero cells AND the homogeneous tau=0 splice point.
        tm = sub['tier_mean'].fillna(0)
        ts = sub['tier_std'].fillna(0)
        sub = sub[(ts > 0) | ((ts == 0) & (tm == 0))]
        sub = sub.dropna(subset=['p_current_i', 'p_best_static_i'])
        if len(sub) == 0:
            print(f"  {sid:3} {label}: (no firm data)")
            continue
        sub['theta_rel_i'] = ((sub['p_current_i'] - sub['p_best_static_i'])
                              / sub['p_current_i'])
        sub['tier_bin'] = sub['tier_i'].apply(lambda t: _bin_tau(t, 7))

        g = sub.groupby('tier_bin')['theta_rel_i']
        bins = sorted(g.groups.keys())
        means = g.mean().reindex(bins)
        sems  = g.sem().reindex(bins)
        counts = g.size().reindex(bins)

        ax.errorbar(bins, means.values, yerr=1.96 * sems.fillna(0).values,
                    fmt='o-', lw=1.6, ms=7, capsize=3, color=color,
                    alpha=0.95, label=label)
        plotted_bins.update(bins)
        plotted = True

        # Per-series diagnostics.
        cells = (sub.groupby(['tier_mean', 'tier_std']).size()
                    .reset_index(name='n')
                    .sort_values(['tier_mean', 'tier_std']))
        print(f"  {sid} {label} -- N firm-rows = {len(sub)}")
        for _, row in cells.iterrows():
            print(f"        bar_tau = {row['tier_mean']:g}, "
                  f"std = {row['tier_std']:g}: n = {int(row['n']):>5}")
        for tau in bins:
            tlab = (r'>=7' if tau == 7 else f'{tau}')
            print(f"        tau_i = {tlab:>3}: "
                  f"n = {int(counts.loc[tau]):>5}  "
                  f"mean theta_rel = {means.loc[tau]:.6f}")

    if not plotted:
        ax.text(0.5, 0.5, '(no series data)',
                transform=ax.transAxes, ha='center', va='center', color='gray')
        return

    ax.set_xlabel(r'Firm visibility $\tau_i$')
    ax.set_ylabel(r'Mean cost gap $\theta_{i,T}$')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=2))
    bins_sorted = sorted(plotted_bins)
    ax.set_xticks(bins_sorted)
    ax.set_xticklabels(
        [(r'$\geq 7$' if b == 7 else str(b)) for b in bins_sorted])
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', lw=0.6, alpha=0.3)
    # Title and legend handled by main() (consistent with panels a/b).


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=None,
                   help=f'Visibility CSVs dir (default: {DEFAULT_VIS_DIR}).')
    p.add_argument('--cost_dir', default=None,
                   help=f'Cost-gap CSVs dir (default: {DEFAULT_COST_DIR}).')
    p.add_argument('--output', default=None,
                   help='Output PNG path '
                        '(default: <data_dir>/figure_visibility.png).')
    args = p.parse_args()

    # Unicode-safe console on Windows.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    visibility_dir = args.data_dir or DEFAULT_VIS_DIR
    cost_dir = args.cost_dir or DEFAULT_COST_DIR
    out_path = args.output or os.path.join(visibility_dir, 'figure_visibility.png')

    vis_df = load_visibility(visibility_dir)
    cost_trials, cost_firms = load_cost(cost_dir)

    plt.rcParams.update({
        'font.size':        11,
        'axes.titlesize':   12,
        'axes.labelsize':   11,
        'xtick.labelsize':  10,
        'ytick.labelsize':  10,
        'legend.fontsize':  9.5,
    })

    # Layout: two gridspecs so (a) and (b) sit tight (shared y-axis,
    # near-zero wspace), while (c) is set off with a wider gap on the right.
    fig = plt.figure(figsize=(13.0, 4.2))
    gs_left  = fig.add_gridspec(1, 2, left=0.060, right=0.620,
                                bottom=0.18, top=0.90, wspace=0.05)
    gs_right = fig.add_gridspec(1, 1, left=0.720, right=0.985,
                                bottom=0.18, top=0.90)
    ax_a = fig.add_subplot(gs_left[0, 0])
    ax_b = fig.add_subplot(gs_left[0, 1], sharey=ax_a)
    ax_c = fig.add_subplot(gs_right[0, 0])
    axes = [ax_a, ax_b, ax_c]

    # (a) Homogeneous tau
    plot_visibility_panel(axes[0], vis_df, tau_mode='homo',
                          title=r'(a) Homo. visibility ($\tau_i = \tau$)',
                          x_label=r'Tier visibility $\tau$',
                          series_ids=['R1', 'R2', 'R3'])
    axes[0].set_ylabel('Fraction of trials converging (solid)\n'
                       'or in periodic regime (dashed)')

    # (b) Heterogeneous tau (lognormal, std = mean), spliced at tau=0
    plot_visibility_panel(axes[1], vis_df, tau_mode='hetero',
                          title=r'(b) Hetero. visibility '
                                r'($\tau_i \sim \mathrm{Poisson}(\bar\tau)$)',
                          x_label=r'Mean tier visibility $\bar{\tau}$',
                          series_ids=['R1', 'R2', 'R3'])
    # Remove y-tick labels on the central panel (shares y with (a)).
    axes[1].tick_params(labelleft=False)
    axes[1].set_ylabel('')

    # (c) Per-firm relative static cost gap, opC hetero-tau only
    axes[2].set_title(r'(c) Final cost gap per firm $\theta_{i,T}$', loc='left')
    plot_perfirm_cost_panel(axes[2], cost_firms)

    # Single legend on panel (a), showing only the solid (converged) entries
    # since the dashed = cycled mapping is in the y-axis label.
    h_all, l_all = axes[0].get_legend_handles_labels()
    kept_h, kept_l = [], []
    for h, l in zip(h_all, l_all):
        if ' -- converged' in l:
            kept_h.append(h)
            kept_l.append(l.replace(' -- converged', ''))
    if kept_h:
        axes[0].legend(kept_h, kept_l, loc='center right', framealpha=0.95)

    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\nWrote {out_path}")


if __name__ == '__main__':
    main()
