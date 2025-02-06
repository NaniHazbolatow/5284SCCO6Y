import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_wave(wave_data, x, timesteps):
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
    plt.show()

def plotting_snapshots(wave_data, x, timesteps, snapshot_arr):
    plt.title('Intermediate Time Steps of the 1D Wave Equation', fontsize=15)
    for i in snapshot_arr:
        plt.plot(x, wave_data[i, :], label=f't = {i}')
    plt.xlabel('x', fontsize=14)
    plt.ylabel('ψ(x,t)', fontsize=14)
    plt.legend()
    plt.show()

def calculate_wave(L, c, N, dt, timesteps, initial_cond, animate=False, plot_snapshots=False, snapshot_arr=None):
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

    if animate:
        animate_wave(wave_data, x, timesteps)
    if plot_snapshots:
        plotting_snapshots(wave_data, x, timesteps, snapshot_arr)

    return wave_data

L = 1
c = 1
N = 500
dt = 0.001
timesteps = 1000
initial_cond = 3
snapshot_arr = [0, 100, 200, 300, 400]
wave_data = calculate_wave(L, c, N, dt, timesteps, initial_cond, animate=True, plot_snapshots=True, snapshot_arr=snapshot_arr)

