import numpy as np


N = 5

arr = np.zeros((N, N))

di = np.diag_indices(N)

di2 = ((di[0] + 1)[:N - 1], np.flip(di[1])[:N - 1])

arr[di2] = np.arange(N - 1) + 1

print(arr)