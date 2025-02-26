import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit, prange


def animate_gray_scott(num_frames, time, N, dx, dt, params):
    """Generates an animation of a Gray-Scott model with the given parameters.

    Args:
        num_frames (int): Number of frames in the animation, equally spaced over simulation time
        time (int): Number of time steps of the simulation
        N (int): grid size
        dx (int): width of each cell on the grid
        dt (float): time increment
        params (list): parameters for: D_u, D_v, f, and k

    Returns:
        Matplotlib Animation
    """    
    _, frames_U = gray_scott_grid(time, N, dx, dt, params, save_frames=True, num_frames=num_frames)

    vmin = np.min(frames_U)
    vmax = np.max(frames_U)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(frames_U[0], extent=[0, N, 0, N], vmin=vmin, vmax=vmax, cmap='inferno')
    ax.set_xlabel('x', fontsize=17)
    ax.set_ylabel('y', fontsize=17)
    ax.set_title('Animation of Species U', fontsize=16)
    cbar = fig.colorbar(im)
    cbar.set_label('Concentration', fontsize=15)
    plt.tight_layout()

    def update(frame):
        im.set_array(frames_U[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=num_frames, interval=50, blit=True)
    plt.close(fig)
    return ani


def plot_gray_scott(times, N, dx, dt, params):
    """Plots the evolution of the Gray-Scott model with three time steps.

    Args:
        times (list): the three time steps that are plotted
        N (int): grid size
        dx (int): width of each cell on the grid
        dt (float): time increment
        params (list): parameters for: D_u, D_v, f, and k
    """    
    if len(times) != 3:
        raise ValueError('Please pick three time values.')

    D_u, D_v, f, k = params
    final_grid_1, _ = gray_scott_grid(times[0], N, dx, dt, params)
    final_grid_2, _ = gray_scott_grid(times[1], N, dx, dt, params)
    final_grid_3, _ = gray_scott_grid(times[2], N, dx, dt, params)

    plt.figure(figsize=(18, 5), dpi=300)
    plt.suptitle(fr'Concentration of U, $D_u = {{{D_u}}}$, $D_v = {{{D_v}}}$, $f = {{{f}}}$, $k = {{{k}}}$', fontsize=18)

    plt.subplot(1, 3, 1)
    plt.title(fr'$t = {{{times[0]}}}$', fontsize=16)
    plt.imshow(final_grid_1[:, :, 0], extent=[0, N, 0, N], cmap='inferno')
    plt.xlabel('x', fontsize=17)
    plt.ylabel('y', fontsize=17)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Concentration U', fontsize=16)

    plt.subplot(1, 3, 2)
    plt.title(fr'$t = {{{times[1]}}}$', fontsize=16)
    plt.imshow(final_grid_2[:, :, 0], extent=[0, N, 0, N], cmap='inferno')
    plt.xlabel('x', fontsize=17)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Concentration U', fontsize=16)

    plt.subplot(1, 3, 3)
    plt.title(fr'$t = {{{times[2]}}}$', fontsize=16)
    plt.imshow(final_grid_3[:, :, 0], extent=[0, N, 0, N], cmap='inferno')
    plt.xlabel('x', fontsize=17)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label('Concentration U', fontsize=16)

    plt.tight_layout()
    plt.show()


@njit(parallel=True)
def update_grid(grid, N, dx, dt, D_u, D_v, f, k):
    """Updates the grid for each time step using Numba parallisation.

    Args:
        grid (ndarray): grid of the previous time step
        N (int): grid size
        dx (int): width of each cell on the grid
        dt (float): time increment
        D_u (float): diffusion constant for species U
        D_v (float): diffusion constant for species V
        f (float): rate of supplementation of species U
        k (float): rate constant

    Returns:
        ndarray: grid at the new time step.
    """    
    new_grid = np.empty_like(grid)

    for i in prange(N):
        for j in range(N):
            D_u_star = D_u/(dx**2) * (grid[(i+1)%N, j, 0] + grid[(i-1)%N, j, 0] + grid[i, (j+1)%N, 0] + grid[i, (j-1)%N, 0] - 4*grid[i, j, 0])
            D_v_star = D_v/(dx**2) * (grid[(i+1)%N, j, 1] + grid[(i-1)%N, j, 1] + grid[i, (j+1)%N, 1] + grid[i, (j-1)%N, 1] - 4*grid[i, j, 1])

            new_grid[i, j, 0] = grid[i, j, 0] + dt * (D_u_star - grid[i, j, 0]*(grid[i, j, 1]**2) + f * (1 - grid[i, j, 0]))
            new_grid[i, j, 1] = grid[i, j, 1] + dt * (D_v_star + grid[i, j, 0]*(grid[i, j, 1]**2) - (f + k) * grid[i, j, 1])
    
    return new_grid


@njit
def gray_scott_grid(time, N, dx, dt, params, save_frames=False, num_frames=100):
    """Sets the initial conditions for the Gray-Scott simulation and evolves the grid. There is an option to save intermediate time steps.

    Args:
        time (int): Number of time steps of the simulation
        N (int): grid size
        dx (int): width of each cell on the grid
        dt (float): time increment
        params (list): parameters for: D_u, D_v, f, and k
        save_frames (bool, optional): Option to save intermediate time steps. Defaults to False.
        num_frames (int, optional): Number of intermediate time steps to save. Defaults to 100.

    Returns:
        ndarray: the grid at the final time step and an array of intermediate time steps.
    """    
    D_u, D_v, f, k = params

    np.random.seed(1)
    grid = np.zeros((N, N, 2))
    grid[:, :, 0] = 1
    start_square, end_square = int((N/2)-(0.1*N)), int((N/2)+(0.1*N))
    noise = np.random.normal(0, 0.02, (end_square - start_square, end_square - start_square))
    grid[start_square:end_square, start_square:end_square, 0] = 0.5 + noise
    grid[start_square:end_square, start_square:end_square, 1] = 0.25 + noise

    num_time_steps = int(time / dt)

    if save_frames:
        times_to_save = np.round(np.linspace(0, num_time_steps - 1, num_frames)).astype(np.int32)
        all_frames_U = np.zeros((num_frames, N, N))

    frame_index = 0
    for t in range(num_time_steps):
        grid = update_grid(grid, N, dx, dt, D_u, D_v, f, k)

        if save_frames and t in times_to_save:
            all_frames_U[frame_index] = grid[:, :, 0].copy()
            frame_index += 1

    return grid, all_frames_U
