import subprocess
import sys


def command(n, cc, AiSi_spread, ntw, iteration):
    return [
        sys.executable,
        "run.py",
        "--exp-type", "ts",
        "--nb-rounds", "50",
        "--nb-firms", str(n),
        "--cc", str(cc),
        "--sigma-w", "0",
        "--sigma-z", "0",
        "--sigma-b", "0",
        "--sigma-a", "0",
        "--aisi-spread", str(AiSi_spread),
        "--network-type", ntw,
        "--exp-name", f"no_anticipation_same_rewiring_order",  # Include iteration to keep caches separate
        "--anticipation-mode", "no_anticipation",  # Options: full, partial, no_anticipation
        "--tier", "10",
        "--export-initntw"  # Export init_ntw experiment data
    ]


n = 20
cc = 4
AiSi_spread = 0.1
for n in [20, 50]:
    for x in range(20):  # 20 different initial networks
        subprocess.run(command(n, cc, AiSi_spread, "new_tech", x))
        for y in range(49):
            print(n, cc, AiSi_spread, x, y)
            subprocess.run(command(n, cc, AiSi_spread, "same_all", x))
            # subprocess.run(command(n, cc, AiSi_spread, "same_tech_new_init", x))
