import numpy as np


N = 10

arr = np.rot90(np.diagflat(np.arange(N - 1, 0, -1), -1))

print(arr)