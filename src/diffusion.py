
import numba
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

@numba.jit(nopython=True, parallel=True)
def solve_laplace(grid, sinks, omega=1.7, tol=1e-5, max_iter=10000):
    """Optimized Successive Over-Relaxation (SOR) solver for Laplace equation."""
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
            break  # Convergence reached

    return grid  # Return updated grid


class DLA2D:
    def __init__(self, grid: tuple, eta: float):
        self.nutrient_grid = np.zeros(shape=grid)
        self.cluster_grid = np.zeros_like(self.nutrient_grid)
        self.nutrient_grid[-1, :] = 1

        seed_position = grid[0] // 2

        self.cluster_grid[0, seed_position] = 1
        self.nutrient_grid[0, seed_position] = 0 # Create a sink

        self.eta = eta
        self.termination = False
        self.termination_step = -1
        
    def update_nutrient_grid(self, omega=1.5, tol=1e-5, max_iter=10000):
        sinks = self.cluster_grid == 1  
        self.nutrient_grid = solve_laplace(self.nutrient_grid, sinks, omega=omega, tol=tol, max_iter=max_iter)
    

    def get_growth_candidates(self):
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

    def growth(self, growth_steps, plot_interval):
        for step in range(growth_steps):
            if self.termination:
                self.termination_step = step + 1
                print(f"Termination at step {step + 1} with {self.eta}")
                if plot_interval > 0:
                    self.plot_state(step+1)
                break

            self.update_nutrient_grid()
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