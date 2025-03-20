import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def leapfrog_harmonic_oscillator(m, k, x0, v0, dt, T, driving_force=None):
    """
    Leapfrog integrator for a harmonic oscillator.
    Optional driving_force parameter is a tuple (amplitude, frequency)
    """

    # Calculate the number of steps and create the time array
    steps = int(T/dt)
    t = np.linspace(0, T, steps+1)

    # Create arrays to store the position and velocity
    x = np.zeros(steps+1)
    v = np.zeros(steps+1)
    
    # Set the initial conditions
    x[0] = x0
    if driving_force is None:
        v[0] = v0 + 0.5 * (-k/m) * x0 * dt
    else:
        # Driven harmonic oscillator
        amplitude, frequency = driving_force
        v[0] = v0 + 0.5 * dt * ((-k/m) * x0 + amplitude/m * np.sin(frequency * 0))

    # Implement the leapfrog algorithm
    for i in range(steps):
        x[i+1] = x[i] + v[i] * dt
        if driving_force is None:
            v[i+1] = v[i] - (k/m) * x[i+1] * dt
        else:
            v[i+1] = v[i] + dt * ((-k/m) * x[i+1] + amplitude/m * np.sin(frequency * t[i+1]))
    
    # Calculate the energy
    energy = 0.5 * m * v**2 + 0.5 * k * x**2

    return x, v, energy, t

def plot_harmonic_oscillator(Ks, Xs, Vs, t):
    fig, axs = plt.subplots(2, 1, figsize=(12, 5))
    for i in range(len(Xs)):
        axs[0].plot(t, Xs[i], label=f'K={Ks[i]}')
        axs[1].plot(t, Vs[i], label=f'K={Ks[i]}')
    axs[0].set_ylabel('Position', fontsize=17)
    axs[1].set_ylabel('Velocity', fontsize=17)
    axs[1].set_xlabel('Time', fontsize=17)

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', fontsize=15)
    #give proper title to the figure
    title="One-Dimensional Harmonic Oscillator using Leapfrog for Different Spring Constants"
    fig.suptitle(f"{title}", fontsize=19)
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=13)

    plt.show()

# Define harmonic oscillator for RK45
def harmonic_oscillator_system(t, y, m, k):
    """
    Returns the derivatives for the harmonic oscillator system.
    y[0] is position, y[1] is velocity
    """
    x, v = y

    # Update the derivatives
    dv_dt = -k/m * x
    dx_dt = v

    return [dx_dt, dv_dt]

def rk45_harmonic_oscillator(m, k, x0, v0, dt, T):
    """
    RK45 integration of a harmonic oscillator for comparison
    """
    # Time points for output
    t_eval = np.linspace(0, T, int(T/dt)+1)
    
    # Initial state
    y0 = [x0, v0]

    # Solve the system
    solution = solve_ivp(
        lambda t, y: harmonic_oscillator_system(t, y, m, k),
        [0, T], y0, method='RK45', t_eval=t_eval)
    
    # Extract position and velocity
    x = solution.y[0]
    v = solution.y[1]
    
    # Calculate energy
    energy = 0.5 * m * v**2 + 0.5 * k * x**2
    
    return x, v, energy, solution.t

def compare_energy_conservation(m, k, x0, v0, dt, T):
    """
    Compare energy conservation between Leapfrog and RK45
    """
    # Run both methods
    _, _, leap_energy, leap_t = leapfrog_harmonic_oscillator(m, k, x0, v0, dt, T)
    _, _, rk45_energy, rk45_t = rk45_harmonic_oscillator(m, k, x0, v0, dt, T)

    # Calculate moving averages over time
    window = 500
    leap_energy = np.convolve(leap_energy, np.ones(window)/window, mode='valid')
    leap_t = leap_t[window//2:len(leap_energy) + window//2]
    rk45_energy = np.convolve(rk45_energy, np.ones(window)/window, mode='valid')
    rk45_t = rk45_t[window//2:len(rk45_energy) + window//2]
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    # Energy plot
    plt.plot(leap_t, leap_energy, label='Leapfrog')
    plt.plot(rk45_t, rk45_energy, '--', label='RK45')
    plt.xlabel('Time', fontsize=17)
    plt.ylabel('Energy', fontsize=17)
    plt.legend()
    plt.title('Energy Conservation Comparison (Leapfrog vs. RK45)', fontsize=19)
    plt.tick_params(axis='both', labelsize=13)
    plt.tight_layout()
    plt.show()


def driven_oscillator_analysis(m, k, x0, v0, dt, T, driving_amplitudes, frequency_ratios):
    """
    Analyze and plot the driven oscillator for various frequency ratios
    """
    # Calculate natural frequency
    natural_freq = np.sqrt(k/m)
    
    # Create phase plots for different frequency ratios
    plt.figure(figsize=(15, 10))
    
    for i, amp in enumerate(driving_amplitudes):
        for j, ratio in enumerate(frequency_ratios):
            # Calculate driving frequency
            driving_freq = ratio * natural_freq
            
            # Run simulation
            x, v, _, t = leapfrog_harmonic_oscillator(
                m, k, x0, v0, dt, T, 
                driving_force=(amp, driving_freq)
            )
            
            # Create phase plot (discarding transient period)
            start_idx = int(len(t) * 0.5)  # Discard first half as transient
            
            plt_idx = i * len(frequency_ratios) + j + 1
            plt.subplot(len(driving_amplitudes), len(frequency_ratios), plt_idx)
            plt.plot(x[start_idx:], v[start_idx:])
            plt.xlabel('Position', fontsize=17)
            plt.ylabel('Velocity',fontsize=17)
            plt.title(f'A={amp}, ω/ω₀={ratio:.2f}', fontsize=18)
            plt.tick_params(axis='both', labelsize=13)
            plt.grid(True)
            
            # Add markers to show direction
            idx = np.linspace(start_idx, len(t)-1, 20).astype(int)
            plt.plot(x[idx], v[idx], 'r.', markersize=5)
    
    plt.tight_layout()
    plt.suptitle('Phase Plots for Driven Harmonic Oscillator', fontsize=19)
    plt.subplots_adjust(top=0.92)
    plt.show()

def animate_driven_oscillator(m, k, x0, v0, dt, T, driving_force):
    """
    Create an animation of the driven oscillator
    """
    # Run the simulation
    x, v, _, t = leapfrog_harmonic_oscillator(m, k, x0, v0, dt, T, driving_force)
    
    # Create the figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Position vs time plot
    line1, = ax1.plot([], [], 'b-', lw=2, label='Position')
    point1, = ax1.plot([], [], 'ro', ms=8)
    ax1.set_xlim(0, T)
    ax1.set_ylim(min(x)*1.2, max(x)*1.2)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Position', fontsize=12)
    ax1.set_title('Position vs Time', fontsize=14)
    ax1.grid(True)
    ax1.legend(fontsize=10)
    
    # Phase space plot
    line2, = ax2.plot([], [], 'g-', lw=2, label='Phase Space')
    point2, = ax2.plot([], [], 'ro', ms=8)
    ax2.set_xlim(min(x)*1.2, max(x)*1.2)
    ax2.set_ylim(min(v)*1.2, max(v)*1.2)
    ax2.set_xlabel('Position', fontsize=12)
    ax2.set_ylabel('Velocity', fontsize=12)
    ax2.set_title('Phase Space', fontsize=14)
    ax2.grid(True)
    ax2.legend(fontsize=10)
    
    plt.tight_layout()
    
    # Add overall title with driving force parameters
    amp, freq = driving_force
    plt.suptitle(f'Driven Harmonic Oscillator (A={amp}, ω={freq:.2f})', fontsize=16, y=1.05)
    
    def init():
        line1.set_data([], [])
        point1.set_data([], [])
        line2.set_data([], [])
        point2.set_data([], [])
        return line1, point1, line2, point2
    
    def animate(i):
        idx = i * 5  # Adjust speed by changing this multiplier
        if idx >= len(t):
            idx = len(t) - 1
            
        line1.set_data(t[:idx], x[:idx])
        point1.set_data([t[idx]], [x[idx]])
        
        line2.set_data(x[:idx], v[:idx])
        point2.set_data([x[idx]], [v[idx]])
        
        return line1, point1, line2, point2
    
    anim = FuncAnimation(fig, animate, init_func=init, 
                        frames=len(t)//5, interval=20, blit=True)
    
    plt.close()  # Prevent display of static figure in Jupyter
    return anim

if __name__ == '__main__':
    # Part 1
    m = 1
    x0 = 1
    v0 = 0
    dt = 0.01
    T = 20

    # Solve the harmonic oscillator for different values of k
    Ks = [1, 2, 4]
    Xs = np.zeros((len(Ks), int(T/dt)+1))
    Vs = np.zeros((len(Ks), int(T/dt)+1))

    for k in Ks:
        x, v, energy, t = leapfrog_harmonic_oscillator(m, k, x0, v0, dt, T)
        Xs[Ks.index(k)] = x
        Vs[Ks.index(k)] = v

    plot_harmonic_oscillator(Ks, Xs, Vs, t)

    # Part 2
    k = 1
    T = 400
    compare_energy_conservation(m, k, x0, v0, dt, T)

    # Part 3    
    # Different driving amplitudes
    driving_amplitudes = [0.5, 2.0]

    # Different frequency ratios
    frequency_ratios = [0.8, 1.0, 1.2]


    natural_freq = np.sqrt(k/m)
    resonant_idx = np.argmin(np.abs(np.array(frequency_ratios) - 1.0))
    resonant_ratio = frequency_ratios[resonant_idx]    
    animation = animate_driven_oscillator(m, k, x0, v0, dt, T, 
                                driving_force=(driving_amplitudes[0], resonant_ratio*natural_freq))
    
    HTML(animation.to_jshtml())

