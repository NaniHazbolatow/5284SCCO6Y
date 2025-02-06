import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc
from numba import njit

def analytical_sol(time, x, D, sum_max=50):
    if time == 0:
        sol = np.zeros(len(x))
        sol[-1] = 1
        return sol
    
    sol = np.zeros(len(x))
    for i in range(sum_max):
        sol += erfc((1 - x + 2 * i)/(2 * np.sqrt(D*time))) - erfc((1 + x + 2 * i)/(2 * np.sqrt(D*time)))
    return sol

@njit
def calculate_grid(grid, time, N, D, dx, dt):
    for _ in range(int(time/dt)):
        new_grid = np.zeros((N, N))
        new_grid[0, :] = 1
        for i in range(1, N-1):
            for j in range(N):
                new_grid[i, j] = grid[i, j] + dt*D/(dx**2) * (grid[i+1, j] + grid[i-1, j] + grid[i, (j+1)%N] + grid[i, (j-1)%N] - 4*grid[i, j])

        grid[:] = new_grid
    return grid

def analytical_vs_experimental(test_times, N, dt, dx, D):
    grid = np.zeros((N, N))
    grid[0, :] = 1
    y = np.linspace(0, 1, N)
    ana_sol_arr = np.zeros((len(test_times), N))
    exp_sol_arr = np.zeros((len(test_times), N))
    for i, t in enumerate(test_times):
        ana_sol = analytical_sol(t, y, D, 50)
        ana_sol_arr[i, :] = ana_sol

        exp_grid = calculate_grid(grid, t, N, D, dx, dt)
        exp_sol_arr[i, :] = np.flip(exp_grid[:, 0])

    for i in range(len(ana_sol_arr)):
        if i == 0:
            plt.plot(y, ana_sol_arr[i, :], color='black', ls='dashed', zorder=2.5, label='Analytical Solutions')
        plt.plot(y, ana_sol_arr[i, :], color='black', ls='dashed', zorder=2.5)
        plt.plot(y, exp_sol_arr[i, :], label=f't = {test_times[i]}')
    plt.title('Analytical vs. Experimental Concentration', fontsize=15)
    plt.xlabel('x', fontsize=14)
    plt.ylabel('c(x)', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()

N = 100
dt = 0.00001
dx = 1/N
D = 1
test_times = [0, 0.001, 0.01, 0.1, 1]

if 4*dt*D/(dx**2) <= 1:
    print('Stable')
else:
    raise ValueError('Not stable, reduce dt')

analytical_vs_experimental(test_times, N, dt, dx, D)
