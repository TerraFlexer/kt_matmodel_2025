import pyopencl as cl
import numpy as np
import matplotlib.pyplot as plt
from time import time


G = 6.67430e-11

def velocity_verlet_opencl(masses, positions, velocities, dt, steps):
    N = len(masses)
    traj = np.zeros((N, 2, steps), dtype=np.float32)
    traj[:, :, 0] = positions
    pos = positions.astype(np.float32).copy()
    vel = velocities.astype(np.float32).copy()
    masses = masses.astype(np.float32).copy()

    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    mf = cl.mem_flags
    pos_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=pos)
    vel_buf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=vel)
    masses_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=masses)

    acc_buf = cl.Buffer(ctx, mf.READ_WRITE, pos.nbytes)

    kernel_code = """
    #define G 6.67430e-11f

    __kernel void compute_acc(
        int N,
        __global const float *masses,
        __global const float2 *pos,
        __global float2 *acc)
    {
        int i = get_global_id(0);
        float2 ai = (float2)(0.0f, 0.0f);
        for (int j = 0; j < N; j++) {
            if (i != j) {
                float2 r = pos[j] - pos[i];
                float dist = length(r) + 1e-10f;
                ai += G * masses[j] * r / (dist * dist * dist);
            }
        }
        acc[i] = ai;
    }

    __kernel void update_pos_vel(
        int N,
        float dt,
        __global float2 *pos,
        __global float2 *vel,
        __global float2 *acc,
        __global const float2 *new_acc)
    {
        int i = get_global_id(0);
        pos[i] += vel[i] * dt + 0.5f * acc[i] * dt * dt;
        vel[i] += 0.5f * (acc[i] + new_acc[i]) * dt;
    }
    """

    prg = cl.Program(ctx, kernel_code).build()

    float2 = np.dtype([('x', np.float32), ('y', np.float32)])
    pos = pos.view(np.float32).reshape(-1, 2)
    vel = vel.view(np.float32).reshape(-1, 2)

    for t in range(1, steps):
        prg.compute_acc(queue, (N,), None, np.int32(N), masses_buf, pos_buf, acc_buf)
        acc = np.empty_like(pos)
        cl.enqueue_copy(queue, acc, acc_buf)

        prg.update_pos_vel(queue, (N,), None, np.int32(N), np.float32(dt), pos_buf, vel_buf, acc_buf, acc_buf)
        cl.enqueue_copy(queue, pos, pos_buf)
        traj[:, :, t] = pos

    return traj

'''def initialize_solar_system():
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

N = 400
steps = 500
dt = 0.01

positions = np.random.rand(N, 2).astype(np.float32)
velocities = np.random.rand(N, 2).astype(np.float32)
masses = np.random.rand(N).astype(np.float32)


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
    
traj = velocity_verlet_opencl(masses, pos0, vel0, dt, steps)

plot_trajectories(
        [traj],
        ['opencl']
    )

np.save('traj_opencl.npy', traj)'''

N = 100
masses = np.random.rand(N) * 1e20
positions = np.random.rand(N, 2) * 1e11
velocities = np.random.rand(N, 2) * 1e3
dt = 0.01
steps = 500

t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time1 = time() - t0
    
t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time2 = time() - t0
    
t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time3 = time() - t0
    
print('N = 100', (opencl_time1 + opencl_time2 + opencl_time3) / 3)
    
N = 200
masses = np.random.rand(N) * 1e20
positions = np.random.rand(N, 2) * 1e11
velocities = np.random.rand(N, 2) * 1e3
dt = 0.01
steps = 500

t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time1 = time() - t0
    
t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time2 = time() - t0
    
t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time3 = time() - t0
    
print('N = 200', (opencl_time1 + opencl_time2 + opencl_time3) / 3)
    
N = 400
masses = np.random.rand(N) * 1e20
positions = np.random.rand(N, 2) * 1e11
velocities = np.random.rand(N, 2) * 1e3
dt = 0.01
steps = 500

t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time1 = time() - t0
    
t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time2 = time() - t0
    
t0 = time()
traj = velocity_verlet_opencl(masses, positions, velocities, dt, steps)
opencl_time3 = time() - t0
    
print('N = 400', (opencl_time1 + opencl_time2 + opencl_time3) / 3)

#N = 100 0.9291419982910156
#N = 200 0.9740900993347168
#N = 400 0.9830666383107504