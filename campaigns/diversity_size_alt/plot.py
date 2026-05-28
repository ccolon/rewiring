"""Companion plotter for the alt diversity-scaling campaign (one panel,
six operating points overlaid).

Reads every *.csv in this script's folder. For each of the six SERIES
defined below, aggregates `same_tech_dif_init` diversity values across
tech matrices (one diversity value per tech matrix, by construction)
and reports the mean +/- 95% normal-approximation CI (mean +/- 1.96 SEM).

Writes:
    figure_diversity_size_alt.{pdf,png}
    diversity_size_alt_data.csv   (long-format summary)

Console: per-cell summary + WARN flag if non-convergence rate > 5%.

Run from anywhere:
    python results/diversity_size_alt/plot_diversity_size_alt.py
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
# Tracked code in campaigns/diversity_size_alt/; CSVs + output figure live in
# the mirroring results/diversity_size_alt/ (gitignored).
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, 'results', 'diversity_size_alt')


# Each series = (tag, b_short, cc, aisi, sw, z_short, kappa,
#                label, color, marker, linestyle).
# The (b_short, cc, aisi, sw, z_short, kappa) tuple is the *unique key*
# that selects rows in the loaded CSVs.
SERIES = [
    ('HRS_dz010',        'unif:0.9:1.1', 4, 0.0,   0.0,  'unif:0.9:1.1',
     1, r'HRS, $\Delta_z = 0.1$',                            'C0', 'o', '-'),
    ('IRS_dz030',        'hom:1.1',      4, 0.0,   0.0,  'unif:0.7:1.3',
     1, r'IRS, $\Delta_z = 0.3$',                            'C3', '^', '-'),
    ('DRS_dz030',        'hom:0.9',      4, 0.0,   0.0,  'unif:0.7:1.3',
     1, r'DRS, $\Delta_z = 0.3$',                            'C2', 's', '-'),
    ('HRS_sw010_dz050',  'unif:0.9:1.1', 4, 0.0,   0.1,  'unif:0.5:1.5',
     1, r'HRS, $\sigma_w = 0.1$, $\Delta_z = 0.5$',          'C1', 'D', '-'),
    ('HRS_dA0005_dz010', 'unif:0.9:1.1', 4, 0.005, 0.0,  'unif:0.9:1.1',
     1, r'HRS, $\Delta_A = 0.005$, $\Delta_z = 0.1$',         'C4', 'p', '-'),
    ('HRS_dA005_k4',     'unif:0.9:1.1', 4, 0.05,  0.0,  'hom:1.0',
     4, r'HRS, $\Delta_A = 0.05$',              'C5', 'X', '--'),
]

N_VALUES = [10, 20, 50, 100, 200, 500]

CSV_OUT_COLS = ['tag', 'n', 'b', 'cprime', 'aisi', 'sigma_w', 'z',
                'kappa', 'nu_mean', 'nu_ci_low', 'nu_ci_high',
                'nonconv_rate', 'n_runs']


def _cfg_str(s):
    """Compact 'hom:V' or 'unif:LO:HI' from a JSON dict (diversity_study output)."""
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


def load_data(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, '*divsize.csv')))
    if not paths:
        raise FileNotFoundError(f"No CSV files in {data_dir!r}")
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        # Only diversity_study-format rows have these columns.
        if 'series' not in d.columns or 'diversity' not in d.columns:
            continue
        dfs.append(d)
    if not dfs:
        raise RuntimeError(f"No diversity-format CSVs in {data_dir}")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['a_short'] = df['a_config'].apply(_cfg_str)
    df['b_short'] = df['b_config'].apply(_cfg_str)
    df['z_short'] = df['z_config'].apply(_cfg_str)
    # All series share these constraints; per-series knobs filtered below.
    df = df[
        (df['series'] == 'same_tech_dif_init') &
        (df['mode'] == 'full') &
        (df['c'] == 4) &
        (df['a_short'] == 'hom:0.5')
    ].copy()
    print(f"Loaded {len(paths)} CSVs -> {len(df)} dif_init rows from {data_dir}")
    return df


def cell_summary(df, n, b_short, cc, aisi, sw, z_short, kappa):
    sub = df[(df['n'] == n) &
             (df['b_short'] == b_short) &
             (df['cc'] == cc) &
             np.isclose(df['aisi_spread'].astype(float), aisi, atol=1e-9) &
             np.isclose(df['sigma_w'].astype(float),    sw,    atol=1e-9) &
             (df['z_short'] == z_short) &
             (df['max_swaps'] == kappa)]
    if len(sub) == 0:
        return None
    vals = sub['diversity'].astype(float).to_numpy()
    nonconv = (1.0 - sub['frac_converged'].astype(float)).mean()
    mu = float(np.mean(vals))
    if len(vals) > 1:
        sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
    else:
        sem = 0.0
    return {
        'mean': mu,
        'ci_low':  max(0.0, mu - 1.96 * sem),
        'ci_high': min(1.0, mu + 1.96 * sem),
        'nonconv_rate': float(nonconv),
        'n_runs': int(len(vals)),
    }


def aggregate(df):
    rows = []
    lookup = {}
    for (tag, b_short, cc, aisi, sw, z_short, kappa,
         label, color, marker, ls) in SERIES:
        for n in N_VALUES:
            s = cell_summary(df, n, b_short, cc, aisi, sw, z_short, kappa)
            if s is None:
                continue
            rec = {
                'tag': tag, 'n': n,
                'b': b_short, 'cprime': cc,
                'aisi': aisi, 'sigma_w': sw, 'z': z_short, 'kappa': kappa,
                'nu_mean':      s['mean'],
                'nu_ci_low':    s['ci_low'],
                'nu_ci_high':   s['ci_high'],
                'nonconv_rate': s['nonconv_rate'],
                'n_runs':       s['n_runs'],
            }
            rows.append(rec)
            lookup[(tag, n)] = rec
    return rows, lookup


def render_figure(lookup, save_path):
    plt.rcParams.update({
        'font.size':       11,
        'axes.titlesize':  12,
        'axes.labelsize':  11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9,
    })
    fig, ax = plt.subplots(figsize=(7.5, 4.6), constrained_layout=True)

    # Plot all series, partitioned by kappa for the two legend groups.
    k1_handles, k1_labels = [], []
    k4_handles, k4_labels = [], []
    for (tag, _b, _cc, _ai, _sw, _z, kappa,
         label, color, marker, ls) in SERIES:
        ns, means, ci_lo, ci_hi = [], [], [], []
        for n in N_VALUES:
            rec = lookup.get((tag, n))
            if rec is None:
                continue
            ns.append(n)
            means.append(rec['nu_mean'])
            ci_lo.append(rec['nu_ci_low'])
            ci_hi.append(rec['nu_ci_high'])
        if not ns:
            continue
        ns = np.array(ns)
        means = np.array(means)
        err_lo = means - np.array(ci_lo)
        err_hi = np.array(ci_hi) - means
        eb = ax.errorbar(ns, means, yerr=[err_lo, err_hi],
                         fmt=marker, ls=ls, color=color, lw=1.6, ms=7,
                         capsize=3, label=label, alpha=0.95)
        if kappa == 4:
            k4_handles.append(eb); k4_labels.append(label)
        else:
            k1_handles.append(eb); k1_labels.append(label)

    ax.set_xscale('log')
    ax.set_xticks(N_VALUES)
    ax.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
    ax.set_xlabel(r'Number of firms $n$')
    ax.set_ylabel(r'Diversity $\nu_{\mathcal{P}}$ (%)')
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', lw=0.6, alpha=0.4)

    # Two stacked legends in the top-left, each with its own title.
    # bbox_to_anchor y for leg2 is set so the second legend sits just below
    # the first; tune if you change font sizes or add/remove series.
    def _left_align(leg):
        # Title and entries flush with the left edge (matplotlib defaults
        # centre the title inside the legend's bounding box).
        leg._legend_box.align = "left"
        leg.get_title().set_ha("left")

    if k1_handles:
        leg1 = ax.legend(k1_handles, k1_labels,
                         loc='upper left',
                         bbox_to_anchor=(0.02, 0.98),
                         title=r'One-at-a-time rewiring ($\kappa = 1$)',
                         framealpha=0.95, title_fontsize=9.5)
        _left_align(leg1)
        ax.add_artist(leg1)
    if k4_handles:
        leg2 = ax.legend(k4_handles, k4_labels,
                         loc='upper left',
                         bbox_to_anchor=(0.02, 0.62),
                         title=r'Unlimited combinations ($\kappa = 4$)',
                         framealpha=0.95, title_fontsize=9.5)
        _left_align(leg2)

    pdf_path = save_path
    png_path = os.path.splitext(save_path)[0] + '.png'
    fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=None,
                   help=f"Default: {DEFAULT_DATA_DIR}.")
    p.add_argument('--output', default=None,
                   help='Output PDF path (default: '
                        '<data_dir>/figure_diversity_size_alt.pdf)')
    p.add_argument('--csv_output', default=None,
                   help='Long-format summary CSV path (default: '
                        '<data_dir>/diversity_size_alt_data.csv)')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    save_path = args.output or os.path.join(data_dir,
                                             'figure_diversity_size_alt.pdf')
    csv_path = args.csv_output or os.path.join(data_dir,
                                                'diversity_size_alt_data.csv')

    df = load_data(data_dir)
    rows, lookup = aggregate(df)

    print()
    print("=== Per-cell summary ===")
    any_warn = False
    for rec in rows:
        warn = ''
        if rec['nonconv_rate'] > 0.05:
            warn = '  WARN (>5%)'
            any_warn = True
        print(f"  {rec['tag']:18}  n={rec['n']:>3}  "
              f"nu={100*rec['nu_mean']:>5.1f}%  "
              f"CI=[{100*rec['nu_ci_low']:>5.1f}%, "
              f"{100*rec['nu_ci_high']:>5.1f}%]  "
              f"nonconv={100*rec['nonconv_rate']:>5.1f}%  "
              f"n_runs={rec['n_runs']:>3d}{warn}")
    if any_warn:
        print("\n  ** Some cells have >5% non-convergence; mention in caption.")

    out = pd.DataFrame(rows, columns=CSV_OUT_COLS)
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    render_figure(lookup, save_path)


if __name__ == '__main__':
    main()
