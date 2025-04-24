from multiprocessing import Pool, cpu_count
import numpy as np
from time import time

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


if __name__ == '__main__':
    N = 100
    masses = np.random.rand(N) * 1e20
    positions = np.random.rand(N, 2) * 1e11
    velocities = np.random.rand(N, 2) * 1e3
    dt = 0.01
    steps = 100

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
    steps = 100

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
    steps = 100

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
    
    print('N = 400', (parallel_time1 + parallel_time2 + parallel_time3) / 3)
    
#N = 100 1.3344839413960774
#N = 200 3.587730646133423
#N = 400 12.913119951883951