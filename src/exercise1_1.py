import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_wave(wave_data, x, timesteps):
    """Makes an animation of the 1D wave equation

    Args:
        wave_data (array): each row contains one time step of the wave
        x (array): x-axis data for plot
        timesteps (int): number of time steps 
    """    
    fig, ax = plt.subplots()
    line, = ax.plot(x, wave_data[0], 'b')
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('ψ(x,t)', fontsize=14)
    ax.set_title('1D Wave Equation Animation', fontsize=15)

    def update(frame):
        line.set_ydata(wave_data[frame])
        return line,

    ani = animation.FuncAnimation(fig, update, frames=timesteps, interval=10, blit=True)
    # plt.show()
    return ani

def plotting_snapshots(wave_data, x, snapshot_arr):
    """Plots some snapshots of wave animation for selected time steps

    Args:
        wave_data (array): contains one time step in each row
        x (array): x-axis data
        snapshot_arr (list): list of time stamps to be plotted
    """    
    plt.title('Intermediate Time Steps of the 1D Wave Equation', fontsize=15)
    for i in snapshot_arr:
        plt.plot(x, wave_data[i, :], label=f't = {i}')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('ψ(x,t)', fontsize=14)
    plt.legend()
    plt.show()

def calculate_wave(L, c, N, dt, timesteps, initial_cond):
    """Calculates the wave equation using the central difference method.

    Args:
        L (int): length of the string
        c (int): constant of wave equation
        N (int): number of intervals
        dt (float): time increment 
        timesteps (int): number of time steps
        initial_cond (int): one of the three initial conditions mentioned in the assignment
        animate (bool, optional): choice to animate. Defaults to False.
        plot_snapshots (bool, optional): choice to plot intermediate time steps. Defaults to False.
        snapshot_arr (list, optional): select which time stamps get plotted ,Defaults to None.

    Returns:
        array: 2D array containing one time step in each row
    """    
    dx = L / (N-1)
    C2 = (c*dt/dx)**2
    x = np.linspace(0, L, N)
    wave_data = np.zeros((timesteps, N))

    if initial_cond == 1:
        wave_data[0, 1:N-1] = np.sin(2*np.pi*x[1:N-1])
    elif initial_cond == 2:
        wave_data[0, 1:N-1] = np.sin(5*np.pi*x[1:N-1])
    elif initial_cond == 3:
        wave_data[0, np.where((x > 1/5) & (x < 2/5))] = np.sin(5*np.pi*x[np.where((x > 1/5) & (x < 2/5))])
    else:
        raise ValueError('Invalid initial condition')

    # First time step:
    wave_data[1, 1:N-1] = wave_data[0, 1:N-1] - 0.5 * C2 * (wave_data[0, 2:N] - 2*wave_data[0, 1:N-1] + wave_data[0, 0:N-2])

    # General update rule
    for t in range(1, timesteps - 1):
        wave_data[t+1, 1:N-1] = 2 * wave_data[t, 1:N-1] - wave_data[t-1, 1:N-1] + C2 * (wave_data[t, 2:N] - 2*wave_data[t, 1:N-1] + wave_data[t, 0:N-2])

    return wave_data, x

