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
        sol = np.zeros_like(x)
        sol[-1] = 1
        return sol
    
    sol = np.zeros_like(x)
    sqrt_term = (2 * np.sqrt(D*time))
    for i in range(sum_max):
        sol += erfc((1 - x + 2 * i)/sqrt_term) - erfc((1 + x + 2 * i)/sqrt_term)
    return sol

@njit
def calculate_grid(time, N, D, dx, dt, save_intermediate=False, num_frames=100):
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
    grid = np.zeros((N, N))
    grid[0, :] = 1.0
    new_grid = grid.copy()
    num_time_steps = int(time / dt)
    coeff = dt * D / (dx ** 2)

    all_frames = None
    if save_intermediate:
        times_to_save = np.round(np.linspace(0, num_time_steps - 1, num_frames)).astype(np.int32)
        all_frames = np.zeros((num_frames, N, N))
    
    # Precompute for periodic boundaries
    left_indices = np.empty(N, dtype=np.int32)
    right_indices = np.empty(N, dtype=np.int32)
    for j in range(N):
        left_indices[j] = j - 1 if j - 1 >= 0 else N - 1
        right_indices[j] = j + 1 if j + 1 < N else 0

    frame_index = 0
    for t in range(num_time_steps):
        for i in range(1, N - 1):
            for j in range(N):
                new_grid[i, j] = grid[i, j] + coeff * (
                    grid[i + 1, j] + grid[i - 1, j] +
                    grid[i, right_indices[j]] + grid[i, left_indices[j]] -
                    4 * grid[i, j]
                )
        grid[:, :] = new_grid[:, :]
        if save_intermediate and t in times_to_save:
            all_frames[frame_index] = grid.copy()
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
    y = np.linspace(0, 1, N)
    ana_sol_arr = np.zeros((len(test_times), N))
    exp_sol_arr = np.zeros((len(test_times), N))
    for i, t in enumerate(test_times):
        ana_sol = analytical_sol(t, y, D, 50)
        ana_sol_arr[i, :] = ana_sol

        exp_grid, _ = calculate_grid(t, N, D, dx, dt)
        exp_sol_arr[i, :] = np.flip(exp_grid[:, 0])

    plt.figure(figsize=(7, 5), dpi=300)
    for i in range(len(ana_sol_arr)):
        if i == 0:
            plt.plot(y, ana_sol_arr[i, :], color='black', ls='dotted', zorder=2.5, label='Analytical Solutions')
        plt.plot(y, ana_sol_arr[i, :], color='black', ls='dotted', zorder=2.5)
        plt.plot(y, exp_sol_arr[i, :], label=f't = {test_times[i]}')
    plt.title('Analytical vs. Experimental Concentration', fontsize=15)
    plt.xlabel('y', fontsize=16)
    plt.ylabel('c(y)', fontsize=16)
    plt.tick_params(axis='both', labelsize=13)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_heatmaps(times: list, grids: list, titles: list, figsize=(18, 5), dpi=300):
    """
    Plots multiple heatmaps side-by-side.
    
    Args:
        times (list): List of times for labeling.
        grids (list): List of 2D numpy arrays to plot.
        titles (list): Titles for each subplot.
        figsize (tuple): Figure size.
        dpi (int): Dots per inch for the figure.
    """
    n = len(grids)
    fig, axs = plt.subplots(1, n, figsize=figsize, dpi=dpi)
    for i in range(n):
        im = axs[i].imshow(grids[i], extent=[0, 1, 0, 1])
        axs[i].set_xlabel('x', fontsize=17)
        axs[i].set_title(titles[i], fontsize=17)
        axs[i].tick_params(axis='both', labelsize=13)
        cbar = plt.colorbar(im, ax=axs[i])
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label('Concentration', fontsize=16)
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
    _, all_frames = calculate_grid(1, N, D, dx, dt, True, num_frames)

    fig, ax = plt.subplots(figsize=(7, 5))
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
