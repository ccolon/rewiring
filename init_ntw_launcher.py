import subprocess
import sys


def command(n, cc, AiSi_spread, ntw):
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
        "--exp-name", "testee",
        "--tier", "10"
    ]


n = 100
cc = 4
for x in range(10):
    for AiSi_spread in [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        subprocess.run(command(n, cc, AiSi_spread, "new"))
        for y in range(49):
            print(n, cc, AiSi_spread, x, y)
            subprocess.run(command(n, cc, AiSi_spread, "inputed"))
