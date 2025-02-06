import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.special import erfc
from numba import njit

def analytical_sol(time, x, D, sum_max=50):
    """Calculates the analytical solution for the concentration

    Args:
        time (int): time 
        x (array): x-axis points
        D (int): diffusion constant
        sum_max (int, optional): number of elements to sum. Defaults to 50.

    Returns:
        array: concentration
    """    
    if time == 0:
        sol = np.zeros(len(x))
        sol[-1] = 1
        return sol
    
    sol = np.zeros(len(x))
    for i in range(sum_max):
        sol += erfc((1 - x + 2 * i)/(2 * np.sqrt(D*time))) - erfc((1 + x + 2 * i)/(2 * np.sqrt(D*time)))
    return sol

@njit
def calculate_grid(grid, time, N, D, dx, dt, save_intermediate=False, num_frames=100):
    """Evolves the grid from the initial condition using finite difference methods and numba for speed

    Args:
        grid (array): grid with initial condition
        time (float): max time
        N (int): grid size 
        D (int): diffusion constant
        dx (float): witdh of each cell in the grid
        dt (float): time increment
        save_intermediate (bool, optional): option to save intermediate grids for animation. Defaults to False.
        num_frames (int, optional): number of frames for the animation. Defaults to 100.

    Returns:
        arrays: grid contains the final grid at t=t_max and all_frames contains all the intermediate grids for the animation.
    """    
    new_grid = np.zeros((N, N))
    new_grid[0, :] = 1
    num_time_steps = int(time/dt)
    if save_intermediate:
        times_to_save = np.round(np.linspace(0, num_time_steps - 1, num_frames)).astype(np.int32)
        all_frames = np.zeros((num_frames, N, N))

    frame_index = 0
    for t in range(num_time_steps):

        for i in range(1, N-1):
            for j in range(N):
                new_grid[i, j] = grid[i, j] + dt*D/(dx**2) * (grid[i+1, j] + grid[i-1, j] + grid[i, (j+1)%N] + grid[i, (j-1)%N] - 4*grid[i, j])

        grid[:] = new_grid

        if save_intermediate and t in times_to_save:
            all_frames[frame_index] = np.copy(grid)
            frame_index += 1

    return grid, all_frames

def analytical_vs_experimental(test_times, N, dt, dx, D):
    """Compares and plots the experimental and analytical solutions for the concentration at different times.

    Args:
        test_times (list): timesteps to measure concentration
        N (int): grid size
        dt (float): time increment
        dx (float): width of cell in grid
        D (int): diffusion constant
    """    
    grid = np.zeros((N, N))
    grid[0, :] = 1
    y = np.linspace(0, 1, N)
    ana_sol_arr = np.zeros((len(test_times), N))
    exp_sol_arr = np.zeros((len(test_times), N))
    for i, t in enumerate(test_times):
        ana_sol = analytical_sol(t, y, D, 50)
        ana_sol_arr[i, :] = ana_sol

        exp_grid, _ = calculate_grid(grid, t, N, D, dx, dt)
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

def plot_heatmap(test_time, N, dt, dx, D):
    """Plots the 2D domain of the concentration for a selected time

    Args:
        test_time (float): time at which the grid is plotted
        N (int): grid size
        dt (float): time increment
        dx (float): width of cell in grid
        D (int): diffusion constant
    """    
    grid = np.zeros((N, N))
    grid[0, :] = 1

    evolved_grid, _ = calculate_grid(grid, test_time, N, D, dx, dt)

    plt.imshow(evolved_grid, extent=[0, 1, 0, 1])
    plt.xlabel('x', fontsize=14)
    plt.ylabel('y', fontsize=14)
    plt.title(f'Diffusion in 2D at t = {test_time}', fontsize=15)
    plt.colorbar(label='Concentration')
    plt.tight_layout()
    plt.show()

def animate_diffusion(num_frames, N, dt, dx, D):
    """Creates an animation of the diffusion equation in 2D

    Args:
        num_frames (int): number of frames in the animation
        N (int): grid size
        dt (float): time increment
        dx (float): width of cell in grid
        D (int): diffusion constant

    Returns:
        animation
    """    
    grid = np.zeros((N, N))
    grid[0, :] = 1
    _, all_frames = calculate_grid(grid, 1, N, D, dx, dt, True, num_frames)

    fig, ax = plt.subplots()
    im = ax.imshow(all_frames[0], extent=[0, 1, 0, 1])
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.set_title('Animation of 2D diffusion', fontsize=15)
    cbar = fig.colorbar(im)
    cbar.set_label('Concentration', fontsize=14)  
    plt.tight_layout()

    def update(frame):
        im.set_array(all_frames[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    plt.close(fig)
    return ani
