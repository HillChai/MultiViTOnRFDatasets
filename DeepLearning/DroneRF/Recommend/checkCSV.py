import numpy as np

s = np.loadtxt("new_dataset/00000_0.csv", delimiter=',', dtype=np.float32)
print(s.shape)

# python3 checkCSV.py
# (2, 10000000)