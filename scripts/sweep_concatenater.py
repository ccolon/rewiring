import os
import sys
import pandas as pd

folder = sys.argv[1]
name = sys.argv[2]

data_path = os.path.join(folder)
files = [f for f in os.listdir(data_path) if name in f and f.endswith(".csv")]

df = pd.concat([pd.read_csv(os.path.join(data_path, f)) for f in files], ignore_index=True)

df.to_csv(folder+'_'+name+'.csv', index=False)
