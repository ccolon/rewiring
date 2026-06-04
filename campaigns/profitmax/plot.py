"""Single-curve diversity-vs-n figure for the appendix "Profit maximisation"
(fig:profitmax) -- profit-max only.

Reads every *.csv in `results/profitmax/` and plots whichever
mode='full_profitmax' n-cells are present. Cost-min rows in the same
directory (if any) are ignored. This keeps the plot working off partial
data: any (n) value without a profit-max CSV is silently skipped rather
than rendered as a gap.

Calibration (must match launch.sh's `--drs_filter` run):
    a = hom 0.5, b = hom 0.9, z = hom 1.0  (Delta_z = 0),
    c = c' = 4, kappa = 1, sigma_w = 0.05, Delta_A = 0.

For each (mode, n) cell, the diversity ν_P value is one observation per
technology matrix (built from `same_tech_dif_init`). We report mean +/- 1.96 *
SEM across tech matrices, as in `campaigns/diversity_size_alt/plot.py`.

Writes:
    figure_profitmax.{pdf,png}
    profitmax_data.csv         (long-format summary)

Run from anywhere (defaults to results/profitmax/):
    python campaigns/profitmax/plot.py
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
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, 'results', 'profitmax')


# The (mode, n)-grid we expect from launch.sh.
N_VALUES = [10, 20, 50, 100, 200]

# Series spec: (mode_key, label, color, marker, linestyle).
# Profit-max only -- cost-min CSVs in the same directory are ignored.
SERIES = [
    ('full_profitmax', r'Profit maximisation', 'C3', 's', '-'),
]

# Filter targets (must match launch.sh).
A_SHORT_TARGET = 'hom:0.5'
B_SHORT_TARGET = 'hom:0.9'
Z_SHORT_TARGET = 'hom:1.0'
C_TARGET       = 4
CC_TARGET      = 4
MS_TARGET      = 1
AISI_TARGET    = 0.0
SIGMAW_TARGET  = 0.05

CSV_OUT_COLS = ['mode', 'n', 'nu_mean', 'nu_ci_low', 'nu_ci_high',
                'nonconv_rate', 'n_runs']


# -----------------------------------------------------------------------------
# Loading & filtering
# -----------------------------------------------------------------------------

def _cfg_str(s):
    """Compact 'hom:V' or 'unif:LO:HI' from the JSON config column."""
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
        if 'series' not in d.columns or 'diversity' not in d.columns:
            continue
        dfs.append(d)
    if not dfs:
        raise RuntimeError(f"No diversity-format CSVs in {data_dir}")
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['a_short'] = df['a_config'].apply(_cfg_str)
    df['b_short'] = df['b_config'].apply(_cfg_str)
    df['z_short'] = df['z_config'].apply(_cfg_str)
    df = df[
        (df['series'] == 'same_tech_dif_init') &
        (df['c']  == C_TARGET) &
        (df['cc'] == CC_TARGET) &
        (df['a_short'] == A_SHORT_TARGET) &
        (df['b_short'] == B_SHORT_TARGET) &
        (df['z_short'] == Z_SHORT_TARGET) &
        (df['max_swaps'] == MS_TARGET) &
        np.isclose(df['aisi_spread'].astype(float), AISI_TARGET, atol=1e-9) &
        np.isclose(df['sigma_w'].astype(float),    SIGMAW_TARGET, atol=1e-9)
    ].copy()
    print(f"Loaded {len(paths)} CSVs -> {len(df)} matching rows from {data_dir}")
    return df


def cell_summary(df, mode, n):
    sub = df[(df['mode'] == mode) & (df['n'] == n)]
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
        'mean':         mu,
        'ci_low':       max(0.0, mu - 1.96 * sem),
        'ci_high':      min(1.0, mu + 1.96 * sem),
        'nonconv_rate': float(nonconv),
        'n_runs':       int(len(vals)),
    }


def aggregate(df):
    rows = []
    lookup = {}
    for (mode, _label, _color, _marker, _ls) in SERIES:
        for n in N_VALUES:
            s = cell_summary(df, mode, n)
            if s is None:
                continue
            rec = {
                'mode': mode, 'n': n,
                'nu_mean':      s['mean'],
                'nu_ci_low':    s['ci_low'],
                'nu_ci_high':   s['ci_high'],
                'nonconv_rate': s['nonconv_rate'],
                'n_runs':       s['n_runs'],
            }
            rows.append(rec)
            lookup[(mode, n)] = rec
    return rows, lookup


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def render_figure(lookup, save_path):
    plt.rcParams.update({
        'font.size':       11,
        'axes.titlesize':  12,
        'axes.labelsize':  11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)

    for (mode, label, color, marker, ls) in SERIES:
        ns, means, ci_lo, ci_hi = [], [], [], []
        for n in N_VALUES:
            rec = lookup.get((mode, n))
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
        ax.errorbar(ns, means, yerr=[err_lo, err_hi],
                    fmt=marker, ls=ls, color=color, lw=1.8, ms=8,
                    capsize=3, label=label, alpha=0.95)

    ax.set_xscale('log')
    ax.set_xticks(N_VALUES)
    ax.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
    ax.set_xlabel(r'Number of firms $n$')
    ax.set_ylabel(r'Configuration diversity $\nu_{\mathcal{P}}$ (%)')
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(alpha=0.3)
    ax.axhline(0, color='black', lw=0.6, alpha=0.4)
    ax.legend(loc='upper left', framealpha=0.95)

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
                        '<data_dir>/figure_profitmax.pdf)')
    p.add_argument('--csv_output', default=None,
                   help='Long-format summary CSV path (default: '
                        '<data_dir>/profitmax_data.csv)')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    save_path = args.output or os.path.join(data_dir, 'figure_profitmax.pdf')
    csv_path = args.csv_output or os.path.join(data_dir, 'profitmax_data.csv')

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
        print(f"  mode={rec['mode']:14}  n={rec['n']:>3}  "
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
