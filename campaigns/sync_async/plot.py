"""Sync-vs-async plot (manuscript app:sync_async, fig:sync_async).

Reads `sync_async_*.csv` from this script's folder; produces `sync_async.pdf`
in the same folder (plus `sync_async.png` for in-repo previewing).

Single panel: histogram of cosine network distances between asynchronous
fixed points and the deterministic synchronous fixed point of the same
(W_bar, S^(0)) pair, pooled across all (pair, async_idx) trials.

Annotations:
  - Vertical line at d = 0 ("async = sync fixed point").
  - Median and IQR overlay (vertical lines).
  - Console summary suitable for the §4.1.2 main-text mention.

Run from anywhere:
    python results/sync_async/plot_sync_async.py
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# This plotter lives in campaigns/sync_async/ (tracked code) but the CSVs and
# output figure live in the mirroring results/sync_async/ (gitignored).
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, 'results', 'sync_async')


def load(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, 'sync_async_*.csv')))
    if not paths:
        raise FileNotFoundError(f"No sync_async_*.csv files in {data_dir!r}")
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(paths)} CSVs -> {len(df)} rows from {data_dir}")
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=None,
                   help=f"Default: {DEFAULT_DATA_DIR}.")
    p.add_argument('--output', default=None,
                   help="Default: <data_dir>/sync_async.pdf (plus .png).")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    out_pdf = args.output or os.path.join(data_dir, 'sync_async.pdf')
    out_png = os.path.splitext(out_pdf)[0] + '.png'

    df = load(data_dir)
    async_df = df[df['run_kind'] == 'async'].copy()
    if len(async_df) == 0:
        raise RuntimeError("No async rows in the loaded CSVs.")

    d = async_df['distance'].to_numpy(dtype=float)
    n_total = len(d)
    n_zero  = int(np.sum(d <= 1e-12))
    med     = float(np.median(d))
    p25     = float(np.percentile(d, 25))
    p75     = float(np.percentile(d, 75))
    pmean   = float(np.mean(d))
    pmax    = float(np.max(d))

    print()
    print(f"N trials (pair x async_idx): {n_total}")
    print(f"  d == 0 (async = sync fixed point): {n_zero} "
          f"({100 * n_zero / max(n_total, 1):.1f}%)")
    print(f"  median d = {med:.4f}    IQR = [{p25:.4f}, {p75:.4f}]")
    print(f"  mean   d = {pmean:.4f}   max d = {pmax:.4f}")
    print()
    print("Suggested manuscript phrasing:")
    print(f'  "Across {n_total:,} trials, the asynchronous-order randomness '
          f'alone produces a network distance of {med:.3f} '
          f'(IQR: {p25:.3f}--{p75:.3f}) from the deterministic synchronous '
          f'fixed point. Mass at d = 0 ({100*n_zero/max(n_total,1):.1f}% of '
          f'trials) reflects pairs where the order channel is degenerate."')

    # ---- Figure -------------------------------------------------------------
    plt.rcParams.update({
        'font.size':       11,
        'axes.titlesize':  12,
        'axes.labelsize':  11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
    bins = np.linspace(0, max(0.05, pmax * 1.05), 40)
    ax.hist(d, bins=bins, color='C0', alpha=0.85, edgecolor='black',
            linewidth=0.4)

    # Annotations
    ax.axvline(0, color='black', lw=1.2, ls='-')
    ax.text(0, ax.get_ylim()[1] * 0.97,
            r' $\,d{=}0$ (async $=$ sync)',
            ha='left', va='top', fontsize=9,
            bbox=dict(facecolor='white', edgecolor='lightgray',
                      alpha=0.92, pad=2))
    ax.axvline(med, color='C3', lw=1.4, ls='--',
               label=f'median = {med:.3f}')
    ax.axvline(p25, color='C3', lw=1.0, ls=':', alpha=0.7,
               label=f'IQR: [{p25:.3f}, {p75:.3f}]')
    ax.axvline(p75, color='C3', lw=1.0, ls=':', alpha=0.7)

    ax.set_xlabel(r'Cosine distance $d(\mathbb{M}^{\rm async}, '
                  r'\mathbb{M}^{\rm sync})$')
    ax.set_ylabel('Count of trials')
    ax.set_xlim(left=-0.005)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(loc='upper right', framealpha=0.95)

    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    print(f"\nWrote {out_pdf}")
    print(f"Wrote {out_png}")
    plt.close(fig)


if __name__ == '__main__':
    main()
