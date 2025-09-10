import subprocess


def command(n, cc, AiSi_spread, ntw):
    return [
        "C:\\Users\\Celian\\miniforge3\\envs\\rewiring\\python.exe",
        "run.py",
        "ts",
        "50",  # maxRound
        str(n),  # n value
        str(cc),  # cc value
        "0", "0", "0", "0",  # sigma_w, sigma_z, sigma_b, sigma_a
        str(AiSi_spread),  # AiSi_spread
        ntw, "n20_spread", "10"
    ]


n = 20
cc = 4
for x in range(10):
    for AiSi_spread in [0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
        subprocess.run(command(n, cc, AiSi_spread, "new"))
        for y in range(49):
            print(n, cc, AiSi_spread, x, y)
            subprocess.run(command(n, cc, AiSi_spread, "inputed"))
