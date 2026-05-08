"""Two-panel visibility figure for the manuscript.

Layout:
    +---------------------+---------------------+
    | homo tau            | hetero tau          |
    | x = tau             | x = mean tau        |
    | multiple regimes    | AiSi-trap regime only
    +---------------------+---------------------+

Y axis (both panels): two solid lines per series:
    - frac_converged (rises with tau)         -> series base color, solid
    - frac_unstable  (= frac never reaching a fixed point or detected cycle,
                      i.e. truncated at nb_rounds)         -> dashed, same color
The implicit middle (frac_cycled = 1 - converged - unstable) is the residual
gap between the two curves -- visually obvious without a third line.

Series shown (n=100, cc=4):
    Left panel (homo tau, tier_std = 0):
        R1: AiSi-trap        a~U[0.4,0.6], b~U[0.9,1.1], z~U[0.9,1.1], aisi=0.05, sw=0.05
        R2: realistic, no AiSi  same a/b/z, aisi=0, sw=0
        R3: fully hom DRS    a=hom 0.5, b=hom 0.9, z=hom 1.0, aisi=0, sw=0
    Right panel (hetero tau, tier_std > 0):
        R1 only -- the only operating point with hetero coverage.

Reads all CSVs from a directory and concatenates.

Usage:
    python scripts/plot_visibility_2panel.py results/visibility
    python scripts/plot_visibility_2panel.py results/visibility --output paper/fig_visibility.png
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


def cfg_str(s):
    try:
        d = json.loads(s)
        if d['mode'] == 'homogeneous':
            return f"hom:{d['value']}"
        return f"unif:{d['min']}:{d['max']}"
    except Exception:
        return 'NA'


# Series identification keys + plotting style
# (id, label, match_keys, color)
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


def load_data(data_dir):
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
            continue          # not a visibility CSV
        dfs.append(d)
    if not dfs:
        raise RuntimeError(f"No visibility-format CSVs in {data_dir}")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['a_short'] = df['a_config'].apply(cfg_str)
    df['b_short'] = df['b_config'].apply(cfg_str)
    df['z_short'] = df['z_config'].apply(cfg_str)
    print(f"Loaded {len(paths)} CSVs -> {len(df)} rows")
    return df


def select(df, keys):
    return df[(df['a_short'] == keys['a']) &
              (df['b_short'] == keys['b']) &
              (df['z_short'] == keys['z']) &
              (df['n'] == keys['n']) &
              (df['cc'] == keys['cc']) &
              (df['aisi_spread'] == keys['aisi']) &
              (df['sigma_w'] == keys['sw'])]


def regime_fractions(sub):
    """Per tier_mean group, return DataFrame with tau, n, frac_conv, frac_cyc,
    frac_unst, plus normal-approx 95% CIs on each fraction.

    Regime classification (one trial per row):
      'fixed'    : converged == 1     OR  cycle_period == 1
      'cycle'    : cycle_period in {2, ..., MAX}
      'unstable' : everything else (cycle_period missing/empty/NaN -> truncated)
    """
    if len(sub) == 0:
        return pd.DataFrame(columns=['tier_mean', 'n', 'p_fix', 'p_cyc', 'p_unst',
                                      'ci_fix', 'ci_unst'])
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
    # 95% normal-approx CI
    for col in ['p_fix', 'p_cyc', 'p_unst']:
        g[f'ci_{col[2:]}'] = 1.96 * np.sqrt((g[col] * (1 - g[col])) / g['n'].clip(lower=1))
    return g.sort_values('tier_mean')


def plot_panel(ax, df, tau_mode, title, x_label, series_ids):
    """tau_mode: 'homo' or 'hetero'. series_ids: list of SERIES_DEFS ids to plot.

    For tau_mode='hetero', tau=0 is well-defined as identical to the homo
    tau=0 point (std=0 by necessity), so we splice the homo tau=0 trials in
    to anchor each series' left endpoint.
    """
    plotted = False
    for sid, label, keys, color in SERIES_DEFS:
        if sid not in series_ids:
            continue
        sub = select(df, keys)
        if tau_mode == 'homo':
            sub = sub[(sub['tau_mode'] == 'homo') & (sub['tier_std'] == 0)]
        else:
            hetero = sub[(sub['tau_mode'] == 'hetero') & (sub['tier_std'] > 0)]
            if len(hetero) == 0:
                continue
            # Splice in homo tau=0 as the leftmost (mean tau=0) point.
            tau0 = sub[(sub['tau_mode'] == 'homo') & (sub['tier_std'] == 0) &
                       (sub['tier_mean'] == 0)]
            sub = pd.concat([tau0, hetero], ignore_index=True, sort=False)
        if len(sub) == 0:
            continue

        g = regime_fractions(sub)
        x = g['tier_mean'].values

        # Solid line: frac_converged (with CI band)
        ax.plot(x, g['p_fix'], 'o-', color=color, lw=1.8, ms=7,
                label=f'{label}  -- converged')
        ax.fill_between(x, g['p_fix'] - g['ci_fix'], g['p_fix'] + g['ci_fix'],
                        color=color, alpha=0.15, linewidth=0)

        # Dashed line: frac_cycled
        ax.plot(x, g['p_cyc'], 'x--', color=color, lw=1.4, ms=7, alpha=0.85,
                label=f'{label}  -- cycled')
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


def coverage_summary(df):
    print("\n=== Coverage per series and tau_mode ===")
    for sid, label, keys, _ in SERIES_DEFS:
        sub = select(df, keys)
        for mode in ['homo', 'hetero']:
            if mode == 'homo':
                m = sub[(sub['tau_mode'] == 'homo') & (sub['tier_std'] == 0)]
            else:
                m = sub[(sub['tau_mode'] == 'hetero') & (sub['tier_std'] > 0)]
            if len(m) == 0:
                continue
            taus = sorted(m['tier_mean'].unique())
            counts = m.groupby('tier_mean').size().to_dict()
            n_per_tau = ', '.join(f't={int(t)}:{counts[t]}' for t in taus)
            print(f"  {sid} {label[:35]:35} | {mode:6} | tau={taus}  "
                  f"trials/tau: {n_per_tau}")


def _select_with_tau0_splice(df, keys, mode):
    """Same filtering as plot_panel: homo uses tier_std==0; hetero splices the
    homo τ=0 row in as the leftmost point."""
    sub = select(df, keys)
    if mode == 'homo':
        return sub[(sub['tau_mode'] == 'homo') & (sub['tier_std'] == 0)]
    hetero = sub[(sub['tau_mode'] == 'hetero') & (sub['tier_std'] > 0)]
    if len(hetero) == 0:
        return hetero
    tau0 = sub[(sub['tau_mode'] == 'homo') & (sub['tier_std'] == 0) &
               (sub['tier_mean'] == 0)]
    return pd.concat([tau0, hetero], ignore_index=True, sort=False)


def print_unstable_breakdown(df):
    """% of unstable runs per (series, mode, tau)."""
    print("\n=== Unstable fraction "
          "(truncated, no fixed point, no detected cycle) ===")
    for sid, label, keys, _ in SERIES_DEFS:
        for mode_label, mode_filter in [('homo  τ', 'homo'), ('hetero τ', 'hetero')]:
            sub = _select_with_tau0_splice(df, keys, mode_filter)
            if len(sub) == 0:
                continue
            cp = pd.to_numeric(sub['cycle_period'], errors='coerce')
            sub2 = sub.copy()
            sub2['is_fix']  = (sub2['converged'].astype(int) == 1) | (cp == 1)
            sub2['is_cyc']  = (~sub2['is_fix']) & (cp >= 2)
            sub2['is_unst'] = (~sub2['is_fix']) & (~sub2['is_cyc'])
            g = (sub2.groupby('tier_mean')
                       .agg(n=('is_unst', 'size'),
                            p_unst=('is_unst', 'mean'))
                       .reset_index()
                       .sort_values('tier_mean'))
            print(f"\n  {sid} -- {label}  ({mode_label})")
            for _, row in g.iterrows():
                print(f"      τ={int(row['tier_mean']):>2d}:  "
                      f"{100 * row['p_unst']:>5.1f}% unstable  (n={int(row['n'])})")


def print_cycle_period_breakdown(df):
    """For each (series, mode, tau) where cycle_fraction > 0, show the period-k
    distribution among cycled runs."""
    print("\n=== Cycle-period breakdown per (series, τ)  "
          "[shown only when cycle fraction > 0] ===")
    for sid, label, keys, _ in SERIES_DEFS:
        for mode_label, mode_filter in [('homo  τ', 'homo'), ('hetero τ', 'hetero')]:
            sub = _select_with_tau0_splice(df, keys, mode_filter)
            if len(sub) == 0:
                continue
            cp = pd.to_numeric(sub['cycle_period'], errors='coerce')
            sub2 = sub.copy()
            sub2['cp'] = cp
            sub2['is_fix'] = (sub2['converged'].astype(int) == 1) | (cp == 1)
            sub2['is_cyc'] = (~sub2['is_fix']) & (cp >= 2)

            printed_header = False
            for tau, g in sub2.groupby('tier_mean'):
                cyc = g[g['is_cyc']]
                total = len(g)
                if len(cyc) == 0:
                    continue
                if not printed_header:
                    print(f"\n  {sid} -- {label}  ({mode_label})")
                    printed_header = True
                breakdown = (cyc['cp'].astype(int)
                               .value_counts(normalize=True)
                               .sort_index())
                periods_str = ',  '.join(
                    f'k={k}: {100*v:.0f}%' for k, v in breakdown.items()
                )
                print(f"      τ={int(tau):>2d}:  "
                      f"cycled={100*len(cyc)/total:>5.1f}% "
                      f"(n={len(cyc)}/{total}) -- {periods_str}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir', nargs='?', default='results/visibility',
                   help='Directory containing visibility CSV files.')
    p.add_argument('--output', default=None,
                   help='Output PNG path (default: <data_dir>/figure_visibility_2panel.png)')
    args = p.parse_args()

    # Allow τ in console output on Windows.
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    df = load_data(args.data_dir)
    coverage_summary(df)
    print_unstable_breakdown(df)
    print_cycle_period_breakdown(df)

    out_path = args.output or os.path.join(args.data_dir, 'figure_visibility_2panel.png')

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

    # Left: homogeneous tau, all 3 regimes
    plot_panel(axes[0], df, tau_mode='homo',
               title=r'(a) Homogeneous $\tau$',
               x_label=r'Tier visibility $\tau$',
               series_ids=['R1', 'R2', 'R3'])

    # Right: heterogeneous tau, R1 spliced with homo tau=0
    plot_panel(axes[1], df, tau_mode='hetero',
               title=r'(b) Heterogeneous $\tau$ (lognormal, std = mean)',
               x_label=r'Mean tier visibility $\bar{\tau}$',
               series_ids=['R1', 'R2', 'R3'])

    # Y-axis label only on the left panel; right inherits ticks via sharey but
    # we hide the tick LABELS explicitly per the user's spec.
    axes[0].set_ylabel(
        'Fraction of trials converging (solid)\n'
        'or trapped in a limit cycle (dashed)'
    )
    axes[1].set_ylabel('')
    axes[1].tick_params(labelleft=False)

    # Single legend on the middle-right of the figure.
    # Build it from the left panel (which has all 3 series) keeping only the
    # converged-line entries, since the dashed style is documented in the
    # y-axis label.
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
