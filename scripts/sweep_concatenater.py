import os
import sys
import pandas as pd

name = sys.argv[1]

data_path = os.path.join('..', 'results_sweep')
files = [f for f in os.listdir(data_path) if name in f and f.endswith(".csv")]

df = pd.concat([pd.read_csv(os.path.join(data_path, f)) for f in files], ignore_index=True)

df.to_csv('results_'+name+'.csv', index=False)
