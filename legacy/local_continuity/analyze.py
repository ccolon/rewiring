"""Analyze the four-cell continuity test:
   fully hom + aisi=0.05, σ_w=0, n=50, ms in {1, 2, 3, 4=cc}, full mode.
"""
import os
import sys
import pandas as pd

DIR = os.path.dirname(os.path.abspath(__file__))
files = {
    1: "fullyhom_aisi05_ms1.csv",
    2: "fullyhom_aisi05_ms2.csv",
    3: "fullyhom_aisi05_ms3.csv",
    4: "fullyhom_aisi05_msEqCc.csv",
}

print(f"{'ms':>3}  {'series':22}  {'ntech':>5}  {'div_mean':>8}  {'div_min':>7}  {'div_max':>7}  "
      f"{'frac_conv':>9}  {'frac_cyc':>8}  {'rounds':>6}  {'sw/firm':>7}")
print("-" * 110)
for ms in sorted(files):
    p = os.path.join(DIR, files[ms])
    if not os.path.exists(p):
        print(f"  ms={ms}: file not yet written ({files[ms]})")
        continue
    df = pd.read_csv(p)
    for series in ["same_tech_same_init", "same_tech_dif_init"]:
        s = df[df["series"] == series]
        if len(s) == 0:
            continue
        print(f"{ms:>3}  {series:22}  {s['tech_seed'].nunique():>5}  "
              f"{s['diversity'].mean():>8.4f}  {s['diversity'].min():>7.3f}  {s['diversity'].max():>7.3f}  "
              f"{s['frac_converged'].mean():>9.4f}  {s['frac_cycled'].mean():>8.4f}  "
              f"{s['mean_rounds'].mean():>6.2f}  {s['mean_swaps_per_firm'].mean():>7.2f}")
