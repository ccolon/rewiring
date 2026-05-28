"""Two-panel figure for fig:diversity_size (manuscript section 4.2.2).

Panel (a): RTS effect at c' = 4. Diversity vs n for CRS / DRS / HRS.
Panel (b): latent-supplier effect at CRS. Diversity vs n for c' in {2, 4, 8}.

Reads every divsize_*.csv in this script's folder, restricts to the
fig:diversity_size operating point (kappa=1, Delta_A=0.05, sigma_w=0,
z=hom 1.0, a=hom 0.5, c=4), then for each (n, b_config, cc) cell:
    - aggregates the same_tech_dif_init `diversity` values across tech
      matrices (one diversity value per tech matrix, by construction);
    - reports mean and 95% normal-approximation CI (mean +/- 1.96 * SEM);
    - reports per-cell non-convergence rate.

Writes:
    figure_diversity_size.{pdf,png}
    diversity_size_data.csv  (long-format summary, see CSV_OUT_COLS below)

Console: per-cell summary + WARN flag if non-convergence rate > 5%.

Run from anywhere:
    python results/diversity_size/plot_diversity_size.py
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


# Panel definitions: (b_config_short, cc, label, color, marker, panel)
SERIES_PANEL_A = [
    ('hom:1.0',      4, 'CRS ($b_i = 1$)',                  'C0', 'o'),
    ('hom:0.9',      4, 'DRS ($b_i = 0.9$)',                'C2', 's'),
    ('unif:0.9:1.1', 4, r'HRS ($b_i\sim U[0.9, 1.1]$)',     'C3', '^'),
]
SERIES_PANEL_B = [
    ('hom:1.0', 2, r"$c' = 2$",  'C4', 'o'),
    ('hom:1.0', 4, r"$c' = 4$",  'C0', 's'),
    ('hom:1.0', 8, r"$c' = 8$",  'C5', '^'),
]

N_VALUES = [10, 20, 50, 100, 200, 500]

CSV_OUT_COLS = ['panel', 'n', 'b', 'cprime',
                'nu_mean', 'nu_ci_low', 'nu_ci_high',
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
    paths = sorted(glob.glob(os.path.join(data_dir, '*.csv')))
    if not paths:
        raise FileNotFoundError(f"No divsize_*.csv files in {data_dir!r}")
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True, sort=False)
    df['a_short'] = df['a_config'].apply(_cfg_str)
    df['b_short'] = df['b_config'].apply(_cfg_str)
    df['z_short'] = df['z_config'].apply(_cfg_str)
    # Restrict to the operating point of fig:diversity_size.
    df = df[
        (df['series'] == 'same_tech_dif_init') &
        (df['mode'] == 'full') &
        (df['max_swaps'] == 1) &
        np.isclose(df['aisi_spread'].astype(float), 0.05, atol=1e-9) &
        np.isclose(df['sigma_w'].astype(float),    0.0,  atol=1e-9) &
        (df['z_short'] == 'hom:1.0') &
        (df['a_short'] == 'hom:0.5') &
        (df['c'] == 4)
    ].copy()
    print(f"Loaded {len(paths)} CSVs -> {len(df)} dif_init rows from {data_dir}")
    return df


def cell_summary(df, n, b_short, cc):
    """Mean, 95% normal-approx CI, non-conv rate, n_runs for a single cell."""
    sub = df[(df['n'] == n) &
             (df['b_short'] == b_short) &
             (df['cc'] == cc)]
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
    """Return (rows_for_csv, dict_lookup keyed by (panel, b_short, cc, n))."""
    rows = []
    lookup = {}

    def add(panel, b_short, cc, n, summary):
        rec = {
            'panel': panel,
            'n': n,
            'b': b_short,
            'cprime': cc,
            'nu_mean':      summary['mean'],
            'nu_ci_low':    summary['ci_low'],
            'nu_ci_high':   summary['ci_high'],
            'nonconv_rate': summary['nonconv_rate'],
            'n_runs':       summary['n_runs'],
        }
        rows.append(rec)
        lookup[(panel, b_short, cc, n)] = rec

    for b_short, cc, *_ in SERIES_PANEL_A:
        for n in N_VALUES:
            s = cell_summary(df, n, b_short, cc)
            if s is None:
                continue
            add('a', b_short, cc, n, s)
    for b_short, cc, *_ in SERIES_PANEL_B:
        for n in N_VALUES:
            s = cell_summary(df, n, b_short, cc)
            if s is None:
                continue
            add('b', b_short, cc, n, s)

    return rows, lookup


def render_figure(lookup, save_path):
    plt.rcParams.update({
        'font.size':       11,
        'axes.titlesize':  12,
        'axes.labelsize':  11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 9.5,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4),
                             constrained_layout=True, sharey=True)

    for ax, panel, series in [(axes[0], 'a', SERIES_PANEL_A),
                              (axes[1], 'b', SERIES_PANEL_B)]:
        for b_short, cc, label, color, marker in series:
            ns, means, ci_lo, ci_hi = [], [], [], []
            for n in N_VALUES:
                rec = lookup.get((panel, b_short, cc, n))
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
                        fmt=marker + '-', color=color, lw=1.6, ms=7,
                        capsize=3, label=label, alpha=0.95)

        ax.set_xscale('log')
        ax.set_xticks(N_VALUES)
        ax.get_xaxis().set_major_formatter(mtick.ScalarFormatter())
        ax.set_xlabel(r'Number of firms $n$')
        ax.set_ylim(-0.03, 1.03)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(alpha=0.3)
        ax.axhline(0, color='black', lw=0.6, alpha=0.4)

    axes[0].set_ylabel(r'Diversity $\nu_{\mathcal{P}}$ (%)')
    axes[0].set_title(r"(a) Returns-to-scale effect ($c' = 4$)", loc='left')
    axes[1].set_title(r"(b) Latent-supplier effect (CRS)", loc='left')
    axes[0].legend(loc='lower right', framealpha=0.95)
    axes[1].legend(loc='lower right', framealpha=0.95)

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
                   help="Default: this script's folder.")
    p.add_argument('--output', default=None,
                   help='Output PDF path (default: '
                        '<data_dir>/figure_diversity_size.pdf)')
    p.add_argument('--csv_output', default=None,
                   help='Long-format summary CSV path (default: '
                        '<data_dir>/diversity_size_data.csv)')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    data_dir = args.data_dir or SCRIPT_DIR
    save_path = args.output or os.path.join(SCRIPT_DIR,
                                             'figure_diversity_size.pdf')
    csv_path = args.csv_output or os.path.join(SCRIPT_DIR,
                                                'diversity_size_data.csv')

    df = load_data(data_dir)
    rows, lookup = aggregate(df)

    # Console summary
    print()
    print("=== Per-cell summary ===")
    any_warn = False
    for rec in rows:
        warn = ''
        if rec['nonconv_rate'] > 0.05:
            warn = '  WARN (>5%)'
            any_warn = True
        print(f"  panel={rec['panel']}  n={rec['n']:>3}  "
              f"b={rec['b']:14}  c'={rec['cprime']}  "
              f"nu={100*rec['nu_mean']:>5.1f}%  "
              f"CI=[{100*rec['nu_ci_low']:>5.1f}%, "
              f"{100*rec['nu_ci_high']:>5.1f}%]  "
              f"nonconv={100*rec['nonconv_rate']:>5.1f}%  "
              f"n_runs={rec['n_runs']:>3d}{warn}")
    if any_warn:
        print("\n  ** Some cells have >5% non-convergence; mention in caption.")

    # Long-format CSV
    out = pd.DataFrame(rows, columns=CSV_OUT_COLS)
    out.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}")

    render_figure(lookup, save_path)


if __name__ == '__main__':
    main()
