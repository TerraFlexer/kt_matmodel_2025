import numpy as np
import matplotlib.pyplot as plt


def add_gauss_noise(z, perc, N):
    ampl = abs(z.max() - z.min()) * perc
    return z + ampl * np.random.randn(N, N)


def add_poiss_noise(z, perc, N):
    ampl = abs(z.max() - z.min()) * perc
    return z + ampl * np.random.poisson(1, (N, N))


N = 100
Edge = np.pi


x = np.linspace(-Edge, Edge, N, endpoint=False)
y = np.linspace(-Edge, Edge, N, endpoint=False)
Y, X = np.meshgrid(x, y)

Z = np.sin(np.sqrt(X**2 + Y**2))

u1 = add_gauss_noise(Z, 0.05, N)

u2 = add_poiss_noise(Z, 0.05, N)

arr = np.array([[Z, u1, abs(Z - u1)], [Z, u2, abs(Z - u2)]])

arr_names = np.array([['Z', 'u1', '|Z - u1|'], ['Z', 'u2', '|Z - u2|']])

fig, axs = plt.subplots(2, 3, figsize=(20, 10))
for col in range(3):
    for row in range(2):
        ax = axs[row, col]
        #pcm = ax.pcolormesh(arr[row, col])
        pcm = ax.contourf(X, Y, arr[row, col], levels = 20)
        ax.set_title(arr_names[row, col])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_aspect('equal')
        labels = ['-$\pi$', '-Pi / 2', '0', 'Pi / 2']

        ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2])
        ax.set_xticklabels(labels)
        ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2])
        ax.set_yticklabels(labels)
        if (col == 2):
            fig.colorbar(pcm, ax=axs[row, col], shrink=0.6)
    if col == 1:
        fig.colorbar(pcm, ax=axs[:, col], shrink=0.6)

plt.show()