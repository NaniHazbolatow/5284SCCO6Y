import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def gray_scott_grid(time, N, D_u, D_v, f, k, dx, dt):

    grid = np.zeros((N, N, 2))
    grid[:, :, 0] = 0.5
    grid[int((N/2)-(0.05*N)):int((N/2)+(0.05*N)), int((N/2)-(0.05*N)):int((N/2)+(0.05*N)), 1] = 0.25

    new_grid = grid.copy()
    num_time_steps = int(time / dt)

    for t in range(num_time_steps):
        for i in range(N):
            for j in range(N):
                D_u_star = D_u/(dx**2) * (grid[(i+1)%N, j, 0] + grid[(i-1)%N, j, 0] + grid[i, (j+1)%N, 0] + grid[i, (j-1)%N, 0] - 4*grid[i, j, 0])
                D_v_star = D_v/(dx**2) * (grid[(i+1)%N, j, 1] + grid[(i-1)%N, j, 1] + grid[i, (j+1)%N, 1] + grid[i, (j-1)%N, 1] - 4*grid[i, j, 1])

                new_grid[i, j, 0] = grid[i, j, 0] + dt * (D_u_star - grid[i, j, 0]*(grid[i, j, 1]**2) + f * (1 - grid[i, j, 0]))
                new_grid[i, j, 1] = grid[i, j, 1] + dt * (D_v_star + grid[i, j, 0]*(grid[i, j, 1]**2) - (f + k) * grid[i, j, 1])

        grid[:, :, :] = new_grid[:, :, :]

    return grid

total_time = 5000
N = 100
D_u = 0.16
D_v = 0.08
f = 0.035
k = 0.060
dx = 1
dt = 1

final_grid = gray_scott_grid(time=total_time, N=N, D_u=D_u, 
                             D_v=D_v, f=f, k=k, dx=dx, dt=dt)

vmin = np.min(final_grid)
vmax = np.max(final_grid)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Concentration of Species U", fontsize=16)
plt.imshow(final_grid[:, :, 0], extent=[0, N, 0, N], vmin=vmin, vmax=vmax)
plt.xlabel('x', fontsize=17)
plt.ylabel('y', fontsize=17)
cbar = plt.colorbar()
cbar.set_label('Concentration U', fontsize=16)

plt.subplot(1, 2, 2)
plt.title("Concentration of Species V", fontsize=16)
plt.imshow(final_grid[:, :, 1], extent=[0, N, 0, N], vmin=vmin, vmax=vmax)
plt.xlabel('x', fontsize=17)
cbar = plt.colorbar()
cbar.set_label('Concentration V', fontsize=16)

plt.tight_layout()
plt.show()