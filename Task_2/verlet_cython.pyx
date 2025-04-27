# verlet_cython.pyx
import numpy as np
cimport numpy as np
cimport cython

cdef double G = 6.67430e-11

@cython.boundscheck(False)
@cython.wraparound(False)
def velocity_verlet_cython(object masses_obj, object positions_obj, object velocities_obj, double dt, int steps):
    cdef int N = positions_obj.shape[0]

    cdef double[::1] masses = masses_obj
    cdef double[:, ::1] pos = positions_obj.copy()
    cdef double[:, ::1] vel = velocities_obj.copy()
    
    cdef np.ndarray traj = np.zeros((N, 2, steps), dtype=np.float64)
    cdef double[:, :, ::1] traj_mv = traj

    cdef int i, j, t
    cdef double dx, dy, dist

    cdef double[:, ::1] acc = np.zeros((N, 2), dtype=np.float64)
    cdef double[:, ::1] new_acc = np.zeros((N, 2), dtype=np.float64)
    cdef double[:, ::1] tmp

    traj_mv[:, :, 0] = pos

    def compute_acc(double[:, ::1] pos_local, double[:, ::1] acc_local):
        cdef int i, j
        cdef double dx, dy, dist

        for i in range(N):
            acc_local[i, 0] = 0.0
            acc_local[i, 1] = 0.0

        for i in range(N):
            for j in range(N):
                if i != j:
                    dx = pos_local[j, 0] - pos_local[i, 0]
                    dy = pos_local[j, 1] - pos_local[i, 1]
                    dist = (dx * dx + dy * dy)**0.5 + 1e-10
                    acc_local[i, 0] += G * masses[j] * dx / dist**3
                    acc_local[i, 1] += G * masses[j] * dy / dist**3

    compute_acc(pos, acc)

    for t in range(1, steps):
        for i in range(N):
            pos[i, 0] += vel[i, 0] * dt + 0.5 * acc[i, 0] * dt * dt
            pos[i, 1] += vel[i, 1] * dt + 0.5 * acc[i, 1] * dt * dt

        compute_acc(pos, new_acc)

        for i in range(N):
            vel[i, 0] += 0.5 * (acc[i, 0] + new_acc[i, 0]) * dt
            vel[i, 1] += 0.5 * (acc[i, 1] + new_acc[i, 1]) * dt

        tmp = acc
        acc = new_acc
        new_acc = tmp

        traj_mv[:, :, t] = pos

    return traj