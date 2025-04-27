from multiprocessing import Pool, cpu_count
import numpy as np
from time import time
import matplotlib.pyplot as plt


G = 6.67430e-11


def compute_acceleration_parallel(args):
    i, pos, masses = args
    N = len(masses)
    acc_i = np.zeros(2)
    for j in range(N):
        if i != j:
            r = pos[j] - pos[i]
            dist = np.linalg.norm(r) + 1e-10
            acc_i += G * masses[j] * r / dist**3
    return i, acc_i


def compute_acc_parallel(pos, masses, pool):
    N = len(masses)
    results = pool.map(compute_acceleration_parallel, [(i, pos, masses) for i in range(N)])
    acc = np.zeros_like(pos)
    for i, acc_i in results:
        acc[i] = acc_i
    return acc


def velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool):
    N = len(masses)
    traj = np.zeros((N, 2, steps))
    traj[:, :, 0] = positions
    pos = positions.copy()
    vel = velocities.copy()

    acc = compute_acc_parallel(pos, masses, pool)
    for t in range(1, steps):
        pos += vel * dt + 0.5 * acc * dt**2
        new_acc = compute_acc_parallel(pos, masses, pool)
        vel += 0.5 * (acc + new_acc) * dt
        acc = new_acc
        traj[:, :, t] = pos
    return traj


'''if __name__ == '__main__':
    N = 100
    masses = np.random.rand(N) * 1e20
    positions = np.random.rand(N, 2) * 1e11
    velocities = np.random.rand(N, 2) * 1e3
    dt = 0.01
    steps = 500

    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time1 = time() - t0
    
    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time2 = time() - t0
    
    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time3 = time() - t0
    
    print('N = 100', (parallel_time1 + parallel_time2 + parallel_time3) / 3)
    
    N = 200
    masses = np.random.rand(N) * 1e20
    positions = np.random.rand(N, 2) * 1e11
    velocities = np.random.rand(N, 2) * 1e3
    dt = 0.01
    steps = 500

    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time1 = time() - t0
    
    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time2 = time() - t0
    
    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time3 = time() - t0
    
    print('N = 200', (parallel_time1 + parallel_time2 + parallel_time3) / 3)
    
    N = 400
    masses = np.random.rand(N) * 1e20
    positions = np.random.rand(N, 2) * 1e11
    velocities = np.random.rand(N, 2) * 1e3
    dt = 0.01
    steps = 500

    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time1 = time() - t0
    
    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time2 = time() - t0
    
    t0 = time()
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, positions, velocities, dt, steps, pool)
    parallel_time3 = time() - t0
    
    print('N = 400', (parallel_time1 + parallel_time2 + parallel_time3) / 3)'''
    
#N = 100 6.023835817972819
#N = 200 20.049667437871296
#N = 400 69.23185578982036


def initialize_solar_system():
    masses = np.array([
        1.989e30,   # Солнце
        3.30e23,    # Меркурий
        4.87e24,    # Венера
        5.97e24,    # Земля
        6.42e23,    # Марс
        1.90e27,    # Юпитер
        5.68e26,    # Сатурн
        8.68e25,    # Уран
        1.02e26     # Нептун
    ])
    
    positions = np.array([
        [0.0, 0.0],                   # Солнце
        [5.79e10, 0.0],               # Меркурий
        [1.08e11, 0.0],               # Венера
        [1.496e11, 0.0],              # Земля
        [2.28e11, 0.0],               # Марс
        [7.78e11, 0.0],               # Юпитер
        [1.43e12, 0.0],               # Сатурн
        [2.87e12, 0.0],               # Уран
        [4.50e12, 0.0]                # Нептун
    ])
    
    velocities = np.array([
        [0.0, 0.0],                   # Солнце
        [0.0, 47400.0],               # Меркурий
        [0.0, 35000.0],               # Венера
        [0.0, 29780.0],               # Земля
        [0.0, 24100.0],               # Марс
        [0.0, 13070.0],               # Юпитер
        [0.0, 9680.0],                # Сатурн
        [0.0, 6800.0],                # Уран
        [0.0, 5430.0]                 # Нептун
    ])
    
    return masses, positions, velocities


masses, pos0, vel0 = initialize_solar_system()
T = 10 * 365 * 24 * 3600
steps = 3000
dt = T / (steps - 1)
t_eval = np.linspace(0, T, steps)


def plot_trajectories(trajs, labels):
    planet_names = [
        "Солнце", "Меркурий", "Венера", "Земля", "Марс",
        "Юпитер", "Сатурн", "Уран", "Нептун"
    ]
    colors = [
        'orange', 'gray', 'gold', 'blue', 'red',
        'brown', 'khaki', 'lightblue', 'darkblue'
    ]
    
    for traj, method_label in zip(trajs, labels):
        plt.figure(figsize=(8, 8))
        for i in range(traj.shape[0]):
            x = traj[i, 0]
            y = traj[i, 1]
            plt.plot(x, y, label=planet_names[i], color=colors[i % len(colors)])
            plt.text(x[-1], y[-1], planet_names[i], fontsize=8, color=colors[i % len(colors)])
        
        plt.title(f'Траектории тел: {method_label}')
        plt.xlabel('X (м)')
        plt.ylabel('Y (м)')
        plt.legend(loc='upper right', fontsize=8)
        plt.axis('equal')
        plt.grid()
        plt.tight_layout()
        plt.show()

def plot_errors(errors_dict, t_eval):
    plt.figure(figsize=(10, 5))
    for name, err in errors_dict.items():
        plt.plot(t_eval, err, label=name)
    plt.title('Погрешности по сравнению с solve_ivp')
    plt.xlabel('Время (с)')
    plt.ylabel('Ошибка (м)')
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == '__main__':
    with Pool(processes=cpu_count()) as pool:
        traj = velocity_verlet_parallel(masses, pos0, vel0, dt, steps, pool)
        plot_trajectories(
            [traj],
            ['opencl']
        )
        np.save('traj.npy', traj)