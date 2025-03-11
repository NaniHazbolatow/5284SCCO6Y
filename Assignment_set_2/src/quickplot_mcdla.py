import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
from mc_dla import run_single_mc_dla, compute_fractal_dimensions_mc_dla
from tqdm import tqdm

grid_size = 100
iterations = 50000
max_growth_steps = 1000
probs = np.linspace(0.1, 1, 10)

def run_simulation(sticking_prob):
    """
    Run one MC-DLA simulation for the given sticking probability.
    Returns a tuple of (sticking_prob, fractal_dimension).
    """
    steps, grid = run_single_mc_dla(grid_size, max_growth_steps, iterations,
                                    sticking_prob,
                                    show_animation=False,
                                    show_walkers=False,
                                    show_final=False)
    fractal_dimension = compute_fractal_dimensions_mc_dla(grid)
    return sticking_prob, fractal_dimension

if __name__ == '__main__':
    # Prepare a dictionary to store the fractal dimensions for each probability.
    fractal_dims = {p: [] for p in probs}
    
    # Build a list of tasks: for each sticking probability, schedule 25 simulations.
    tasks = [p for p in probs for _ in range(25)]
    
    # Execute the simulations in parallel with progress bar.
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(run_simulation, tasks),
                           total=len(tasks),
                           desc="Running simulations",
                           unit="sim"))
    
    # Group the results by sticking probability.
    for p, fd in results:
        fractal_dims[p].append(fd)
    
    # Plot the fractal dimensions.
    plt.figure(figsize=(7, 5), dpi=300)
    means = [np.mean(fractal_dims[p]) for p in probs]
    stds = [np.std(fractal_dims[p]) for p in probs]
    plt.errorbar(probs, means, yerr=1.96*(stds/np.sqrt(25)),
                 fmt='o', capsize=4, ecolor='grey', color='blue', elinewidth=2)
    
    plt.xlabel('Sticking Probability ($p_s$)', fontsize=15)
    plt.ylabel('Fractal Dimension D', fontsize=15)
    plt.title('Fractal Dimension vs. $p_s$', fontsize=15)
    plt.tick_params(axis='both', labelsize=13)
    plt.tight_layout()
    plt.grid(True, linestyle='--')
    plt.ylim(1.2, 1.8)
    
    #save the plot
    plt.savefig('figures/mc_dla/fractal_dimension_vs_ps.png')