"""Analyze the per-firm rewiring stats produced by
visibility_per_firm_stats_run.py.

Three diagnostics:

  (1) Average number of rewires per firm per round over the first R=10
      rounds, by cell. If a trial stopped before 10 rounds, average over
      `min(10, rounds)`. Output: one number per cell + summary by op_label.

  (2) For limit-cycle trials only: rewire-events in the last k rounds (one
      full period); count distinct firms appearing; report their tau_i values
      as a list. Output: one row per cycling trial.

  (3) For unstable trials (rounds == NB_ROUNDS): mean swaps per firm per
      round over the LAST R=10 rounds, aggregated by cell and by op_label.

Usage:
    python scripts/visibility_per_firm_stats_analysis.py
    python scripts/visibility_per_firm_stats_analysis.py --input <path>.pkl
"""
import argparse
import os
import pickle
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

R = 10
NB_ROUNDS = 50


# -----------------------------------------------------------------------------
# Per-trial round-level counts
# -----------------------------------------------------------------------------

def per_round_swaps(trial):
    """Return a numpy array of length `rounds` with the total swap count in
    each round (rounds 1..R indexed at array positions 0..R-1)."""
    counts = np.zeros(trial['rounds'], dtype=int)
    for ev in trial['rewire_events']:
        r = int(ev['round'])
        if 1 <= r <= len(counts):
            counts[r - 1] += 1
    return counts


def first_R_swaps_per_firm_per_round(trial, r=R):
    """Mean swaps per firm per round over rounds 1..min(r, rounds_run)."""
    rounds_used = min(r, trial['rounds'])
    if rounds_used == 0:
        return np.nan
    swaps = per_round_swaps(trial)[:rounds_used].sum()
    return swaps / (trial['n'] * rounds_used)


def last_R_swaps_per_firm_per_round(trial, r=R):
    """Mean swaps per firm per round over the last min(r, rounds_run) rounds."""
    rounds_used = min(r, trial['rounds'])
    if rounds_used == 0:
        return np.nan
    swaps = per_round_swaps(trial)[-rounds_used:].sum()
    return swaps / (trial['n'] * rounds_used)


# -----------------------------------------------------------------------------
# (1) First-R rounds diagnostic
# -----------------------------------------------------------------------------

def report_first_R(trials):
    print(f"\n=== (1) Rewiring rate -- first {R} rounds (or until convergence) ===")
    print(f"     swaps per firm per round, averaged over the cell's trials\n")
    print(f"  {'cell_id':10}  {'op_label':30}  {'mean':>7}  {'std':>7}  "
          f"{'n_trials':>8}  {'avg_rounds_used':>15}")
    print('  ' + '-' * 92)
    by_cell = {}
    for t in trials:
        by_cell.setdefault(t['cell_id'], []).append(t)
    for cell_id, cell_trials in sorted(by_cell.items()):
        rates = [first_R_swaps_per_firm_per_round(t, R) for t in cell_trials]
        rounds_used = [min(R, t['rounds']) for t in cell_trials]
        op_label = cell_trials[0]['op_label']
        print(f"  {cell_id:10}  {op_label[:30]:30}  "
              f"{np.nanmean(rates):>7.4f}  {np.nanstd(rates):>7.4f}  "
              f"{len(cell_trials):>8}  {np.mean(rounds_used):>15.1f}")


# -----------------------------------------------------------------------------
# (2) Cycling trials: distinct firms per cycle period + tau_i list
# -----------------------------------------------------------------------------

def report_cycle_breakdown(trials):
    print(f"\n=== (2) Limit-cycle trials only:  firms rewiring during "
          f"one full cycle period ===")
    cycled = [t for t in trials
              if isinstance(t.get('cycle_period'), int) and t['cycle_period'] >= 2]
    if len(cycled) == 0:
        print("  (no cycled trials)")
        return
    print(f"  {len(cycled)} cycled trials out of {len(trials)} total\n")

    by_cell = {}
    for t in cycled:
        by_cell.setdefault(t['cell_id'], []).append(t)

    for cell_id, cell_trials in sorted(by_cell.items()):
        op_label = cell_trials[0]['op_label']
        print(f"  {cell_id} ({op_label}) -- {len(cell_trials)} cycled trials:")
        for t in cell_trials:
            k = t['cycle_period']
            r_max = t['rounds']
            r_lo = r_max - k + 1            # last k rounds [r_lo, r_max]
            cycle_firms = set()
            for ev in t['rewire_events']:
                if r_lo <= int(ev['round']) <= r_max:
                    cycle_firms.add(int(ev['firm']))
            tier_arr = np.asarray(t['tier_arr'], dtype=int)
            tau_i_list = sorted(int(tier_arr[f]) for f in cycle_firms)
            print(f"      trial {t['trial_idx']:>2}: cycle_period={k:>2}  "
                  f"n_distinct_firms={len(cycle_firms):>3}  "
                  f"tau_i = {tau_i_list}")


# -----------------------------------------------------------------------------
# (3) Unstable trials: mean swaps per firm per round in last R
# -----------------------------------------------------------------------------

def report_unstable_last_R(trials):
    print(f"\n=== (3) Unstable trials only (rounds == {NB_ROUNDS}): "
          f"mean swaps per firm per round in last {R} rounds ===")
    unstable = [t for t in trials
                if t['rounds'] >= NB_ROUNDS and not t['converged']
                and (t.get('cycle_period') is None or
                     not isinstance(t.get('cycle_period'), int))]
    print(f"  {len(unstable)} unstable trials out of {len(trials)} total\n")
    if len(unstable) == 0:
        return

    # Per cell
    print(f"  {'cell_id':10}  {'op_label':30}  {'n_unstable':>10}  {'mean':>7}  "
          f"{'std':>7}")
    print('  ' + '-' * 75)
    by_cell = {}
    for t in unstable:
        by_cell.setdefault(t['cell_id'], []).append(t)
    for cell_id, cell_trials in sorted(by_cell.items()):
        rates = [last_R_swaps_per_firm_per_round(t, R) for t in cell_trials]
        op_label = cell_trials[0]['op_label']
        print(f"  {cell_id:10}  {op_label[:30]:30}  {len(cell_trials):>10}  "
              f"{np.nanmean(rates):>7.4f}  {np.nanstd(rates):>7.4f}")

    # Aggregated by op_label
    print(f"\n  Aggregated by op_label:")
    print(f"  {'op_label':30}  {'n_unstable':>10}  {'mean':>7}  {'std':>7}")
    print('  ' + '-' * 60)
    by_op = {}
    for t in unstable:
        by_op.setdefault(t['op_label'], []).append(t)
    for op, op_trials in sorted(by_op.items()):
        rates = [last_R_swaps_per_firm_per_round(t, R) for t in op_trials]
        print(f"  {op[:30]:30}  {len(op_trials):>10}  "
              f"{np.nanmean(rates):>7.4f}  {np.nanstd(rates):>7.4f}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default=None,
                   help='Pickle path; default: results/visibility_per_firm_stats.pkl')
    args = p.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    in_path = (args.input or
               os.path.join(REPO_ROOT, 'results', 'visibility_per_firm_stats.pkl'))
    with open(in_path, 'rb') as f:
        trials = pickle.load(f)
    print(f"Loaded {len(trials)} trials from {in_path}")

    # Top-line classification summary
    print(f"\n=== Trial classification ===")
    print(f"  {'cell_id':10}  {'converged':>9}  {'cycled':>6}  {'unstable':>8}  "
          f"{'total':>5}")
    print('  ' + '-' * 50)
    by_cell = {}
    for t in trials:
        by_cell.setdefault(t['cell_id'], []).append(t)
    for cell_id, cell_trials in sorted(by_cell.items()):
        n_total = len(cell_trials)
        n_conv  = sum(1 for t in cell_trials if t['converged'])
        n_cyc   = sum(1 for t in cell_trials
                      if isinstance(t.get('cycle_period'), int)
                      and t['cycle_period'] >= 2)
        n_unst  = n_total - n_conv - n_cyc
        print(f"  {cell_id:10}  {n_conv:>9}  {n_cyc:>6}  {n_unst:>8}  {n_total:>5}")

    report_first_R(trials)
    report_cycle_breakdown(trials)
    report_unstable_last_R(trials)


if __name__ == '__main__':
    main()
