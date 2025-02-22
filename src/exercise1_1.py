import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit

def animate_wave(wave_data, x, timesteps):
    """Makes an animation of the 1D wave equation

    Args:
        wave_data (array): each row contains one time step of the wave
        x (array): x-axis data for plot
        timesteps (int): number of time steps 

    Returns:
        animation
    """    
    fig, ax = plt.subplots(figsize=(7, 5))
    line, = ax.plot(x, wave_data[0], 'b')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x', fontsize=15)
    ax.set_ylabel('ψ(x,t)', fontsize=15)
    ax.set_title('1D Wave Equation Animation', fontsize=15)

    def update(frame):
        line.set_ydata(wave_data[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=10, blit=True)
    plt.close(fig)
    return ani

def plotting_snapshots(wave_data_list, titles, x, snapshot_arr, figsize = None):
    """
    Plots snapshots of the wave data for selected time steps.

    Args:
        wave_data_list: List of 2D arrays, each representing wave data from a different initial condition.
        titles: List of titles for each subplot.
        x: 1D array for x-axis data.
        snapshot_arr: List of time steps to plot.
        figsize: Figure size. Defaults to (6 * number of subplots, 5).
    """
    n_plots = len(wave_data_list)
    if figsize is None:
        figsize = (6 * n_plots, 5)
    fig, axs = plt.subplots(1, n_plots, figsize=figsize, dpi=300)

    # If we want only one plot, make sure this works
    if n_plots == 1:
        axs = [axs]

    for ax, wave_data, title in zip(axs, wave_data_list, titles):
        for t in snapshot_arr:
            ax.plot(x, wave_data[t], label=f't = {t}')
        ax.set_title(title, fontsize=16)
        ax.set_xlabel('x', fontsize=16)
        ax.tick_params(axis='both', labelsize=12)
        ax.legend()
    plt.tight_layout()
    plt.show()

@njit
def calculate_wave(L, c, N, dt, timesteps, initial_cond):
    """Calculates the wave equation using the central difference method.

    Args:
        L (float): length of the string.
        c (float): wave speed constant.
        N (int): number of spatial points.
        dt (float): time increment.
        timesteps (int): number of time steps.
        initial_cond (int): one of the three initial conditions.
            1: psi(x,0) = sin(2πx) for interior points.
            2: psi(x,0) = sin(5πx) for interior points.
            3: psi(x,0) = sin(5πx) for x in (1/5, 2/5), 0 elsewhere.

    Returns:
        tuple: (wave_data, x) where wave_data is a 2D array (time steps × spatial points)
               and x is the spatial grid.
    """
    dx = L / (N - 1)
    C2 = (c * dt / dx) ** 2
    x = np.linspace(0, L, N)
    wave_data = np.zeros((timesteps, N))

    if initial_cond == 1:
        # Initial condition: psi(x,0) = sin(2πx) for interior points.
        wave_data[0, 1:-1] = np.sin(2 * np.pi * x[1:-1])
    elif initial_cond == 2:
        # Initial condition: psi(x,0) = sin(5πx) for interior points.
        wave_data[0, 1:-1] = np.sin(5 * np.pi * x[1:-1])
    elif initial_cond == 3:
        # Initial condition: psi(x,0) = sin(5πx) for 1/5 < x < 2/5, 0 elsewhere.
        mask = (x > 1/5) & (x < 2/5)
        wave_data[0, mask] = np.sin(5 * np.pi * x[mask])
    else:
        raise ValueError("Invalid initial condition")

    # Compute the first time step.
    wave_data[1, 1:-1] = wave_data[0, 1:-1] + 0.5 * C2 * (
        wave_data[0, 2:] - 2 * wave_data[0, 1:-1] + wave_data[0, :-2]
    )

    # General updates
    for t in range(1, timesteps - 1):
        wave_data[t + 1, 1:-1] = (
            2 * wave_data[t, 1:-1]
            - wave_data[t - 1, 1:-1]
            + C2 * (wave_data[t, 2:] - 2 * wave_data[t, 1:-1] + wave_data[t, :-2])
        )

    return wave_data, x

