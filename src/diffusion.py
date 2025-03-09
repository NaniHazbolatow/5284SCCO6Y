
import numba
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

@numba.jit(nopython=True, parallel=True)
def solve_laplace(grid, sinks, omega=1.7, tol=1e-5, max_iter=10000):
    """
    Solve the Laplace equation using Red-Black Successive Over-Relaxation (SOR).
    
    This function solves the steady-state diffusion equation (Laplace equation)
    using an optimized Red-Black SOR method with periodic boundary conditions
    in the horizontal direction.
    
    Parameters:
    -----------
    grid : numpy.ndarray
        Initial concentration grid to be solved
    sinks : numpy.ndarray
        Boolean mask indicating sink locations (True where sinks are present)
    omega : float, optional
        Relaxation parameter (default: 1.7)
    tol : float, optional
        Convergence tolerance (default: 1e-5)
    max_iter : int, optional
        Maximum number of iterations (default: 10000)
        
    Returns:
    --------
    tuple
        (grid, iterations) - The solved grid and number of iterations performed
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
    Diffusion Limited Aggregation (DLA) simulation in 2D with nutrient field.
    
    This class implements a DLA model where growth is influenced by a nutrient
    concentration field that is solved using the Laplace equation. The growth
    probability is proportional to the nutrient concentration raised to the
    power of eta.
    """
    def __init__(self, grid: tuple, eta: float):
        """
        Initialize a new DLA2D simulation.
        
        Parameters:
        -----------
        grid : tuple
            Tuple of (height, width) specifying the grid dimensions
        eta : float
            Exponent controlling the influence of nutrient concentration on growth probability
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
        Update the nutrient concentration field by solving the Laplace equation.
        
        Parameters:
        -----------
        omega : float, optional
            Relaxation parameter for SOR method (default: 1.5)
        tol : float, optional
            Convergence tolerance (default: 1e-5)
        max_iter : int, optional
            Maximum number of iterations (default: 10000)
            
        Returns:
        --------
        int
            Number of iterations performed to reach convergence
        """
        sinks = self.cluster_grid == 1  
        self.nutrient_grid, iter_count = solve_laplace(self.nutrient_grid, sinks, omega=omega, tol=tol, max_iter=max_iter)
        return iter_count

    def get_growth_candidates(self):
        """
        Find all valid growth candidate cells adjacent to the existing cluster.
        
        A cell is a valid growth candidate if it is empty and has at least one
        neighboring cell that is part of the cluster.
        
        Returns:
        --------
        numpy.ndarray
            Array of [row, col] indices for all valid growth candidate cells
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
        Select and add a new cell to the cluster based on nutrient concentration.
        
        This method:
        1. Gets all valid growth candidate cells
        2. Calculates growth probabilities based on nutrient concentration raised to eta power
        3. Randomly selects a cell based on these probabilities
        4. Adds the selected cell to the cluster
        5. Sets the nutrient concentration at that cell to zero (creating a sink)
        6. Checks if the cluster has reached the bottom boundary for termination
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
        Visualize the current state of the DLA simulation.
        
        This method creates a figure showing the nutrient concentration field
        with the cluster overlaid in white.
        
        Parameters:
        -----------
        step : int
            Current simulation step (used for the title)
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
        Run the DLA growth process for a specified number of steps.
        
        This method iteratively updates the nutrient field and adds new cells to the
        cluster until either the specified number of growth steps is reached or
        the cluster reaches the bottom boundary (termination condition).
        
        Parameters:
        -----------
        growth_steps : int
            Maximum number of growth steps to perform
        plot_interval : int
            Interval for plotting the simulation state. If <= 0, no intermediate plots are shown.
        omega : float, optional
            Relaxation parameter for SOR method (default: 1.5)
        tol : float, optional
            Convergence tolerance (default: 1e-5)
        max_iter : int, optional
            Maximum number of iterations for solving Laplace equation (default: 10000)
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
    """Compute the fractal dimension using the box-counting method."""
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
    """Fit a power law to estimate the fractal dimension."""
    log_sizes = np.log(sizes)
    log_counts = np.log(counts)

    slope, _ = np.polyfit(log_sizes, log_counts, 1)
    return -slope  # The fractal dimension D

def compute_fractal_dimensions(eta_values, grid_size=(100, 100), growth_steps=5000):
    """
    Compute fractal dimensions for DLA clusters with different eta values.
    
    This function runs a DLA simulation for each eta value and computes
    the fractal dimension of the resulting cluster.
    
    Parameters:
    -----------
    eta_values : list or array
        List of eta values to test
    grid_size : tuple, optional
        Size of the grid as (height, width) (default: (100, 100))
    growth_steps : int, optional
        Maximum number of growth steps for each simulation (default: 5000)
        
    Returns:
    --------
    tuple
        (eta_values, fractal_dimensions) - The input eta values and corresponding fractal dimensions
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


def stochastic_runs_fd(eta_values, runs, grid_size=(100,100)):
    """
    Perform multiple DLA simulations for each eta value to compute statistical properties.
    
    This function runs multiple DLA simulations for each eta value to calculate
    the mean and standard deviation of the fractal dimension, providing insight
    into the stochastic variability of the growth process.
    
    Parameters:
    -----------
    eta_values : list or array
        List of eta values to test
    runs : int
        Number of simulation runs for each eta value
    grid_size : tuple, optional
        Size of the grid as (height, width) (default: (100, 100))
        
    Returns:
    --------
    tuple
        (eta_values, avg_fractal_dimensions, std_fractal_dimensions) -
        The input eta values, mean fractal dimensions, and standard deviations
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
    Run and visualize multiple DLA simulations with different eta values.
    
    This function creates a side-by-side comparison of DLA clusters grown
    with different eta values, showing both the nutrient field and the
    cluster structure.
    
    Parameters:
    -----------
    grid_size : tuple
        Size of the grid as (height, width)
    etas : list or array
        List of eta values to simulate and compare
    growth_steps : int
        Maximum number of growth steps for each simulation
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
    Records the SOR iteration count for each nutrient update.
    Returns the average iteration count and the list of iteration counts.
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

    