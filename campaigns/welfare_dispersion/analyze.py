"""Welfare-dispersion analyzer: reads `welfare_*.csv` files from this script's
folder and writes `tab_welfare_dispersion.tex` (the manuscript's
tab:welfare_dispersion table).

Per parameter point P0..P4:
  - Statistics restricted to converged trials only.
  - U_star = max U_T across converged trials at this point (per the spec).
  - Three columns reported:
      1. sigma(U_T) / E(U_T)
      2. (U_star - U_T_min) / U_star
      3. (U_star - U_T_median) / U_star

Convergence-rate footnote: counts of non-converged trials per point.

For the U_AA fallback per the design choice "use the last observed sum_p
value as a fallback": U_T is computed from final_prices regardless of
convergence in the study script, so no additional fallback logic is
needed here. The analyzer simply reports whether convergence was
achieved.

Reads any number of welfare_*.csv files in this directory (one per
(point, batch) pair from launch_welfare_dispersion.sh).

Run from anywhere:
    python campaigns/welfare_dispersion/analyze.py
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# This analyzer lives in campaigns/welfare_dispersion/ (tracked code) but the
# CSVs + the output tex table live in the mirroring results/welfare_dispersion/
# (gitignored). 3 dirnames lands at the repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, 'results', 'welfare_dispersion')

POINT_ORDER = ['P0', 'P1', 'P2', 'P3', 'P4']
POINT_DISPLAY = {
    'P0': r"P0 (CRS, $\kappa = c'$, no dispersion)",
    'P1': r"P1 (CRS, $\kappa = 1$, no dispersion)",
    'P2': r"P2 (CRS, $\kappa = 1$, $\Delta_A = 0.05$)",
    'P3': r"P3 (HRS, $\kappa = c'$, no dispersion)",
    'P4': r"P4 (HRS, $\kappa = 1$, compounded)",
}


def load(data_dir):
    paths = sorted(glob.glob(os.path.join(data_dir, 'welfare_*.csv')))
    if not paths:
        raise FileNotFoundError(f"No welfare_*.csv files in {data_dir!r}")
    dfs = [pd.read_csv(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    print(f"Loaded {len(paths)} CSVs -> {len(df)} rows from {data_dir}")
    return df


def per_point_stats(df_point):
    """Return dict with the three reported stats + counts.
    Restricts to converged trials per the spec.
    """
    n_total = len(df_point)
    n_conv = int(df_point['converged'].sum())
    sub = df_point[df_point['converged'] == 1]
    if len(sub) == 0:
        return {'cv': float('nan'),
                'max_min': float('nan'),
                'max_median': float('nan'),
                'n_total': n_total, 'n_conv': 0,
                'U_star': float('nan'),
                'mean': float('nan'), 'std': float('nan'),
                'median': float('nan'), 'min': float('nan')}
    U = sub['U_T'].to_numpy(dtype=float)
    U_star  = float(np.max(U))
    U_min   = float(np.min(U))
    U_med   = float(np.median(U))
    mu      = float(np.mean(U))
    sigma   = float(np.std(U, ddof=1)) if len(U) > 1 else 0.0
    # Coefficient of variation. Note: U_T can be near-zero (e.g. P0 CRS),
    # so we report sigma/|mu| if |mu| > eps, else NaN.
    cv = sigma / abs(mu) if abs(mu) > 1e-12 else float('nan')
    g1 = (U_star - U_min) / U_star if U_star != 0 else float('nan')
    g2 = (U_star - U_med) / U_star if U_star != 0 else float('nan')
    return {'cv': cv, 'max_min': g1, 'max_median': g2,
            'n_total': n_total, 'n_conv': n_conv,
            'U_star': U_star, 'mean': mu, 'std': sigma,
            'median': U_med, 'min': U_min}


def _fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return '---'
    if abs(x) < 1e-4:
        return f"{x:.2e}"
    return f"{x:.4f}"


def render_table(stats_by_point, save_path):
    lines = []
    lines.append(r"\begin{table}[h!]")
    lines.append(r"\centering")
    lines.append(r"\caption{Welfare dispersion across fixed points at five "
                 r"representative parameter points. P0 is the AA-baseline "
                 r"control with a unique fixed point; P1--P4 progressively "
                 r"introduce search-friction and structural multiplicity. "
                 r"Each row aggregates over up to 2{,}500 simulations "
                 r"(50 technology matrices $\times$ 50 initial networks), "
                 r"restricted to converged trials.}")
    lines.append(r"\label{tab:welfare_dispersion}")
    lines.append(r"\begin{tabular}{l c c c}")
    lines.append(r"\toprule")
    lines.append(r"Point & $\sigma(U_T) / \mathrm{E}(U_T)$ "
                 r"& $\frac{U_T^\star - U_T^{\min}}{U_T^\star}$ "
                 r"& $\frac{U^\star - U_T^{\mathrm{median}}}{U^\star}$ \\")
    lines.append(r"\midrule")
    for pid in POINT_ORDER:
        if pid not in stats_by_point:
            lines.append(f"{POINT_DISPLAY[pid]:50} & --- & --- & --- \\\\")
            continue
        s = stats_by_point[pid]
        lines.append(
            f"{POINT_DISPLAY[pid]:50} & "
            f"{_fmt(s['cv'])} & {_fmt(s['max_min'])} & {_fmt(s['max_median'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    with open(save_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {save_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default=None,
                   help=f"Default: {DEFAULT_DATA_DIR}.")
    p.add_argument('--output', default=None,
                   help="Default: <data_dir>/tab_welfare_dispersion.tex.")
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    data_dir = args.data_dir or DEFAULT_DATA_DIR
    out_path = args.output or os.path.join(data_dir,
                                            'tab_welfare_dispersion.tex')

    df = load(data_dir)

    print("\n=== Per-point summary ===")
    stats_by_point = {}
    for pid in POINT_ORDER:
        sub = df[df['point'] == pid]
        if len(sub) == 0:
            print(f"  {pid}: (no data)")
            continue
        s = per_point_stats(sub)
        stats_by_point[pid] = s
        print(
            f"  {pid}: n_total={s['n_total']:>5}  conv={s['n_conv']:>5}  "
            f"({100*s['n_conv']/max(s['n_total'],1):>5.1f}%)  "
            f"U_T mean={s['mean']:>8.4f}  sd={s['std']:>8.4f}  "
            f"median={s['median']:>8.4f}  min={s['min']:>8.4f}  "
            f"U_star={s['U_star']:>8.4f}"
        )

    # Convergence-rate footnote.
    print("\n=== Convergence-rate footnote (>5% non-converged?) ===")
    for pid in POINT_ORDER:
        if pid not in stats_by_point:
            continue
        s = stats_by_point[pid]
        rate = 100 * (s['n_total'] - s['n_conv']) / max(s['n_total'], 1)
        flag = " *" if rate > 5 else ""
        print(f"  {pid}: {rate:.1f}% non-converged{flag}")

    render_table(stats_by_point, out_path)


if __name__ == '__main__':
    main()
