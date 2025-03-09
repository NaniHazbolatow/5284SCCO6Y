
import numba
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

@numba.jit(nopython=True, parallel=True)
def solve_laplace(grid, sinks, omega=1.7, tol=1e-5, max_iter=10000):
    """
    Solves the Laplace equation using Successive Over-Relaxation (SOR) with Red-Black Gauss-Seidel updates.
    
    Parameters:
        grid (np.ndarray): Initial grid with boundary conditions.
        sinks (np.ndarray): Boolean mask indicating sink locations.
        omega (float): Relaxation parameter for SOR.
        tol (float): Convergence tolerance.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        tuple: Updated grid and the number of iterations performed.
    """
    ny, nx = grid.shape
    residual = tol + 1  # Ensure at least one iteration runs

    # Precompute neighbor indices to optimize periodic BC lookups
    west_idx = np.arange(nx) - 1
    east_idx = np.arange(nx) + 1
    west_idx[0] = nx - 1  # Periodic BC left
    east_idx[-1] = 0      # Periodic BC right

    for iteration in range(max_iter):
        residual = 0.0

        # Perform Red-Black Gauss-Seidel update
        for parity in range(2):  # 0 = red, 1 = black
            for i in numba.prange(1, ny - 1):
                for j in range(nx):
                    if (i + j) % 2 != parity or sinks[i, j]:
                        continue  # Skip sinks and enforce Red-Black pattern

                    # Neighboring values
                    north = grid[i - 1, j]
                    south = grid[i + 1, j]
                    west = grid[i, west_idx[j]]
                    east = grid[i, east_idx[j]]

                    # SOR Update
                    old_value = grid[i, j]
                    new_value = (1 - omega) * old_value + omega * (north + south + west + east) / 4.0
                    new_value = max(0.0, new_value)
                    residual = max(residual, abs(new_value - old_value))
                    grid[i, j] = new_value  # In-place update

        if residual < tol:
            # Return both the grid and the number of iterations (iteration+1)
            return grid, iteration + 1

    return grid, max_iter  



class DLA2D:
    """
    Diffusion-Limited Aggregation (DLA) simulation in 2D.
    
    Attributes:
        nutrient_grid (np.ndarray): The grid representing nutrient concentration.
        cluster_grid (np.ndarray): The grid representing the aggregated particles.
        eta (float): Growth bias exponent.
        termination (bool): Indicates whether growth has reached the top boundary.
        termination_step (int): Step at which growth terminated.
    """
    def __init__(self, grid: tuple, eta: float):
        """
        Initializes a DLA simulation on a given grid size.
        
        Parameters:
            grid (tuple): Tuple specifying grid dimensions (ny, nx).
            eta (float): Growth bias exponent.
        """
        self.nutrient_grid = np.zeros(shape=grid)
        self.cluster_grid = np.zeros_like(self.nutrient_grid)
        self.nutrient_grid[-1, :] = 1.0

        seed_position = grid[0] // 2

        self.cluster_grid[0, seed_position] = 1
        self.nutrient_grid[0, seed_position] = 0.0 # Create a sink

        self.eta = eta
        self.termination = False
        self.termination_step = -1
        
    def update_nutrient_grid(self, omega=1.5, tol=1e-5, max_iter=10000):
        """
        Updates the nutrient grid by solving the Laplace equation.
        
        Parameters:
            omega (float): Relaxation parameter for SOR.
            tol (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.
        
        Returns:
            int: Number of iterations performed during the update.
        """
        sinks = self.cluster_grid == 1  
        self.nutrient_grid, iter_count = solve_laplace(self.nutrient_grid, sinks, omega=omega, tol=tol, max_iter=max_iter)
        return iter_count

    def get_growth_candidates(self):
        """
        Identifies potential growth sites adjacent to existing clusters.
        
        Returns:
            np.ndarray: Array of (y, x) coordinates of candidate sites.
        """
        grid = self.cluster_grid

        east = np.roll(grid, shift=-1, axis=1)
        west = np.roll(grid, shift=1, axis=1)
        north = np.roll(grid, shift=1, axis=0)
        south = np.roll(grid, shift=-1, axis=0)

        north[0, :] = 0  # top row invalid for north shift
        south[-1, :] = 0  # bottom row invalid for south shift

        # A cell qualifies if at least one of its four neighbors is part of the cluster.
        neighbor_occupied = (north + south + east + west) > 0

        # Exclude cells that are already part of the cluster.
        candidates = (grid == 0) & neighbor_occupied

        # Return the indices of all growth candidate cells.
        return np.argwhere(candidates)
    
    def choose_growth_candidate(self):
        """
        Chooses a growth site based on the local nutrient concentration and updates the cluster.
        """
        candidate_indices = self.get_growth_candidates()
        nutrient_values = (
            self.nutrient_grid[candidate_indices[:, 0], candidate_indices[:, 1]]
            ** self.eta
        )
        probabilities = nutrient_values / nutrient_values.sum()

        picked_index = candidate_indices[np.random.choice(len(candidate_indices), p=probabilities)]

        if picked_index[0] == self.nutrient_grid.shape[0] - 1:
            self.termination = True

        self.cluster_grid[picked_index[0], picked_index[1]] = 1
        self.nutrient_grid[picked_index[0], picked_index[1]] = 0


    def plot_state(self, step):
        """
        Plots the current state of the simulation.
        
        Parameters:
            step (int): Current step number for labeling the plot.
        """
        plt.figure(figsize=(8, 6))

        # Plot nutrient grid
        plt.imshow(self.nutrient_grid, cmap="inferno", origin="lower")
        plt.colorbar(label="Nutrient concentration")

        # Overlay particle grid with a white mask
        cluster_overlay = np.zeros_like(self.nutrient_grid, dtype=np.float64)  # Ensure correct dtype
        cluster_overlay[self.cluster_grid == 1] = 1  # Mark clusters

        # Ensure alpha mask is float64
        alpha_mask = np.where(self.cluster_grid == 1, 1.0, 0.0).astype(np.float64)

        # Overlay white clusters using grayscale with full opacity on clusters
        plt.imshow(cluster_overlay, cmap="gray", origin="lower", alpha=alpha_mask)

        plt.title(f"Nutrient and Particle Grid with η = {self.eta:.2f}")
        plt.show()

    def growth(self, growth_steps, plot_interval, omega=1.5, tol=1e-5, max_iter=10000):
        """
        Runs the DLA growth process for a given number of steps.
        
        Parameters:
            growth_steps (int): Number of steps to simulate.
            plot_interval (int): Interval at which to plot the grid (-1 for no plots).
            omega (float): Relaxation parameter for SOR.
            tol (float): Convergence tolerance.
            max_iter (int): Maximum number of iterations.
        """
        for step in range(growth_steps):
            # Termination occured in the last step, so offset step by 1
            if self.termination:
                self.termination_step = step 
                print(f"Termination at step {step} with {self.eta}")
                # This will plot the final output
                if plot_interval > 0:
                    self.plot_state(step)
                break

            self.update_nutrient_grid(omega=omega, tol=tol, max_iter=max_iter)
            self.choose_growth_candidate()
            
            if plot_interval > 0 and (step + 1) % plot_interval == 0:
                print(f"Plotting at step {step + 1} with {self.eta}")
                self.plot_state(step + 1)

        if self.termination_step < 0:
            self.termination_step = growth_steps 


def box_counting(grid):
    """
    Compute the fractal dimension using the box-counting method.

    The function divides the given binary grid into progressively smaller squares (boxes) and counts 
    the number of non-empty boxes at each scale.

    Parameters:
        grid (numpy.ndarray): A 2D binary array representing the cluster, where occupied sites are 1.

    Returns:
        tuple:
            - sizes (numpy.ndarray): An array of box sizes used for the analysis.
            - counts (numpy.ndarray): The corresponding count of non-empty boxes at each scale.
    """
    sizes = np.array([2**i for i in range(1, int(np.log2(min(grid.shape))))])  # Ensure divisibility
    counts = []

    for size in sizes:
        reshaped = grid[:grid.shape[0]//size*size, :grid.shape[1]//size*size]  # Trim to be divisible
        reshaped = reshaped.reshape(grid.shape[0]//size, size, grid.shape[1]//size, size)
        non_empty_boxes = np.sum(np.any(reshaped, axis=(1, 3)))
        counts.append(non_empty_boxes)

    counts = np.array(counts)  # Convert to NumPy array

    valid_indices = counts > 0
    sizes = sizes[valid_indices]
    counts = counts[valid_indices]

    return sizes, counts


def fit_fractal_dimension(sizes, counts):
    """
    Fit a power law to estimate the fractal dimension using a linear regression in log-log space.

    Parameters:
        sizes (numpy.ndarray): Array of box sizes.
        counts (numpy.ndarray): Corresponding number of non-empty boxes.

    Returns:
        float: The estimated fractal dimension.
    """
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)

    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return -slope  # The fractal dimension D

def compute_fractal_dimensions(eta_values, grid_size=(100, 100), growth_steps=5000):
    """
    Compute the fractal dimensions for different eta values in a DLA simulation.

    Parameters:
        eta_values (list of float): List of eta values to simulate.
        grid_size (tuple of int, optional): Dimensions of the simulation grid. Default is (100, 100).
        growth_steps (int, optional): Number of growth steps for the DLA process. Default is 5000.

    Returns:
        tuple:
            - eta_values (list of float): The input eta values.
            - fractal_dimensions (list of float): The computed fractal dimensions for each eta.
    """
    fractal_dimensions = []

    for eta in eta_values:
        print(f"Computing fractal dimension for η = {eta:.2f}")
        dla = DLA2D(grid=grid_size, eta=eta)
        dla.growth(growth_steps, plot_interval=-1)  # Grow the cluster

        sizes, counts = box_counting(dla.cluster_grid)
        D = fit_fractal_dimension(sizes, counts)

        fractal_dimensions.append(D)
        print(f"Fractal Dimension for η={eta:.2f}: {D:.3f}")

    return eta_values, fractal_dimensions


# Define eta values and compute fractal dimensions for more runs
def stochastic_runs_fd(eta_values, runs, grid_size=(100,100)):
    """
    Compute the average and standard deviation of fractal dimensions over multiple DLA runs.

    Parameters:
        eta_values (list of float): List of eta values to simulate.
        runs (int): Number of stochastic runs per eta value.
        grid_size (tuple of int, optional): Dimensions of the simulation grid. Default is (100, 100).

    Returns:
        tuple:
            - eta_values (list of float): The input η values.
            - avg_fractal_dimensions (list of float): The average fractal dimension for each eta.
            - std_fractal_dimensions (list of float): The standard deviation of the fractal dimension for each eta.
    """
    avg_fractal_dimensions = []
    std_fractal_dimensions = []

    for eta in eta_values:
        fractal_dimensions = []

        print(f"Computing fractal dimension for η = {eta:.2f} over {runs} runs...")

        for run in range(runs):
            print(f"  Run {run+1}/{runs} for η = {eta:.2f}")
            dla = DLA2D(grid=grid_size, eta=eta)
            dla.growth(growth_steps=10000, plot_interval=-1)  # Run until termination

            sizes, counts = box_counting(dla.cluster_grid)
            D = fit_fractal_dimension(sizes, counts)
            fractal_dimensions.append(D)

        mean_D = np.mean(fractal_dimensions)
        std_D = np.std(fractal_dimensions)

        avg_fractal_dimensions.append(mean_D)
        std_fractal_dimensions.append(std_D)

        print(f"η={eta:.2f}: Mean Fractal Dimension = {mean_D:.3f}, Std = {std_D:.3f}")

    return eta_values, avg_fractal_dimensions, std_fractal_dimensions

def plot_many_dla(grid_size, etas, growth_steps):
    """
    Run and visualize multiple DLA simulations for different η values.

    Parameters:
        grid_size (tuple of int): Dimensions of the simulation grid.
        etas (list of float): List of eta values for simulation.
        growth_steps (int): Number of growth steps for each DLA model.

    Displays:
        A side-by-side plot of the generated DLA clusters for different eta values.
    """
    # Initialize DLA models for different η values
    dla_models = [DLA2D(grid_size, eta) for eta in etas]

    # Run growth process for each model
    for model in dla_models:
        model.growth(growth_steps, plot_interval=0)  # No intermediate plots

    # Plot all results side by side
    fig, axes = plt.subplots(1, len(etas), figsize=(6 * len(etas), 6))

    if len(etas) == 1:
        axes = [axes]  # Ensure axes is iterable for a single subplot

    for ax, model, eta in zip(axes, dla_models, etas):
        ax.imshow(model.cluster_grid, cmap="gray", origin="lower")
        ax.set_title(f"η = {eta}")
        ax.axis("off")

        # Plot nutrient grid
        ax.imshow(model.nutrient_grid, cmap="inferno", origin="lower")

        # Overlay particle grid with a white mask
        cluster_overlay = np.zeros_like(model.nutrient_grid, dtype=np.float64)  # Ensure correct dtype
        cluster_overlay[model.cluster_grid == 1] = 1  # Mark clusters

        # Ensure alpha mask is float64
        alpha_mask = np.where(model.cluster_grid == 1, 1.0, 0.0).astype(np.float64)

        # Overlay white clusters using grayscale with full opacity on clusters
        ax.imshow(cluster_overlay, cmap="gray", origin="lower", alpha=alpha_mask)

        ax.set_title(f"η = {eta:.2f}, Steps = {model.termination_step}")

    plt.tight_layout()
    plt.show()


def growth_iterations(grid_shape, eta, omega, growth_steps=50, tol=1e-5, max_iter=10000):
    """
    Runs a DLA simulation for a given number of growth steps with specified omega and eta.

    Parameters:
        grid_shape (tuple of int): Dimensions of the simulation grid.
        eta (float): Growth parameter affecting adhesion probability.
        omega (float): Relaxation factor for the Successive Over-Relaxation (SOR) method.
        growth_steps (int, optional): Number of steps for the growth process. Default is 50.
        tol (float, optional): Tolerance for convergence in the SOR method. Default is 1e-5.
        max_iter (int, optional): Maximum number of iterations for the SOR method. Default is 10000.

    Returns:
        tuple:
            - avg_iter (float): The average number of iterations per step.
            - iter_counts (list of int): The list of iteration counts per step.
    """
    simulation = DLA2D(grid_shape, eta)
    iter_counts = []
    for step in range(growth_steps):
        iter_count = simulation.update_nutrient_grid(omega=omega, tol=tol, max_iter=max_iter)
        iter_counts.append(iter_count)
        simulation.choose_growth_candidate()
        if simulation.termination:
            print(f"Terminated at step {step + 1} for η = {eta}, ω = {omega}")
            break
    avg_iter = np.mean(iter_counts)
    return avg_iter, iter_counts

# Experiment function that loops over both eta and omega values.
def optimal_omega_eta(grid_shape, eta_values, omega_values, growth_steps=50, tol=1e-5, max_iter=10000):
    """
    For a given grid shape, loops over a range of eta and omega values,
    running the full growth simulation and recording the average SOR iterations.

    Parameters:
        grid_shape (tuple of int): The dimensions of the simulation grid.
        eta_values (list of float): A list of η values to test.
        omega_values (list of float): A list of ω values to test.
        growth_steps (int, optional): The number of growth steps for each simulation. Default is 50.
        tol (float, optional): Convergence tolerance for the SOR method. Default is 1e-5.
        max_iter (int, optional): Maximum number of iterations allowed for the SOR method. Default is 10000.
    
    Returns:
        avg_iters: 2D numpy array of shape (len(eta_values), len(omega_values)) with average iterations.
    """
    avg_iters = np.zeros((len(eta_values), len(omega_values)))
    for i, eta in enumerate(eta_values):
        for j, omega in enumerate(omega_values):
            avg_iter, _ = growth_iterations(
                grid_shape, eta, omega, growth_steps=growth_steps, tol=tol, max_iter=max_iter
            )
            avg_iters[i, j] = avg_iter
            print(f"η: {eta:.3f}, ω: {omega:.3f} -> Average iterations: {avg_iter:.2f}")
    return avg_iters

    
