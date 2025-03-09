import numpy as np
import matplotlib.pyplot as plt

def box_counting(grid):
    """Compute the fractal dimension using the box-counting method."""
    sizes = np.array([2**i for i in range(1, int(np.log2(min(grid.shape))))])
    counts = []
    
    for size in sizes:
        # Trim grid to be divisible by box size
        reshaped = grid[:grid.shape[0]//size*size, :grid.shape[1]//size*size]
        # Reshape to group cells into boxes
        boxes = reshaped.reshape(grid.shape[0]//size, size, grid.shape[1]//size, size)
        # Count boxes that contain any part of the cluster
        non_empty_boxes = np.sum(np.any(boxes, axis=(1, 3)))
        counts.append(non_empty_boxes)
    
    counts = np.array(counts)
    # Only use box sizes where we found cluster cells
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

def spawn_walker(grid, grid_size):
    """
    Spawn a new walker at the top of the grid.
    
    Parameters:
    -----------
    grid : numpy.ndarray
        The current grid state
    grid_size : int
        Size of the grid
        
    Returns:
    --------
    tuple or (None, grid)
        If successful, returns (spawn_position, updated_grid)
        If no available positions, returns (None, grid)
    """
    available_positions = [y for y in range(grid_size) if grid[0, y] == 0]

    if not available_positions:
        return None, grid

    y_pos = available_positions[np.random.randint(0, len(available_positions))]
    spawn_position = [0, y_pos]
    grid[spawn_position[0], spawn_position[1]] = 2
    return spawn_position, grid
    
def update_walker_position(walker, grid, grid_size, s_prob):
    """
    Update the position of a walker on the grid, handling sticking to the cluster
    and movement with periodic boundary conditions.
    
    Parameters:
    -----------
    walker : list
        Current [x, y] position of the walker
    grid : numpy.ndarray
        The current grid state
    grid_size : int
        Size of the grid
    s_prob : float
        Sticking probability (0.0 to 1.0)
        
    Returns:
    --------
    tuple
        (new_position, updated_grid, reached_top)
        - new_position: New walker position or None if walker stuck to cluster
        - updated_grid: Updated grid state
        - reached_top: Boolean indicating if cluster reached the top boundary
    """
    x, y = walker
    
    # First, check for sticking at current position
    adjacent_positions = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]
    for adj_x, adj_y in adjacent_positions:
        # Handle horizontal wrap-around
        if adj_y >= grid_size:
            adj_y = 0
        elif adj_y < 0:
            adj_y = grid_size - 1
            
        # Skip if outside vertical bounds
        if adj_x >= grid_size or adj_x < 0:
            continue
            
        # Check if adjacent to cluster
        if grid[adj_x, adj_y] == 3 and np.random.rand() < s_prob:
            grid[x, y] = 3  # Convert walker to cluster
            # Check if cluster has reached the top boundary
            if x == 0:
                return None, grid, True
            return None, grid, False
    
    # If no sticking occurred, try to move
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
    np.random.shuffle(directions)
    
    for direction in directions:
        new_x = x + direction[0]
        new_y = y + direction[1]
        
        # Handle horizontal wrap-around
        if new_y >= grid_size:
            new_y = 0
        elif new_y < 0:
            new_y = grid_size - 1
        
        # Check if vertical elimination and respawn is needed
        if new_x >= grid_size or new_x < 0:
            grid[x, y] = 0
            new_pos, grid = spawn_walker(grid, grid_size)
            return new_pos, grid, False
        
        # Try to move if the new position is empty
        if grid[new_x, new_y] == 0:
            grid[new_x, new_y] = 2  # Move walker to new position
            grid[x, y] = 0  # Clear old position
            return [new_x, new_y], grid, False
    
    # If no move was possible, stay in place
    return walker, grid, False

def run_single_mc_dla(grid_size, max_growth_steps, iterations=1000, s_prob=0.5, show_animation=False, show_walkers=False, show_final=False):
    """
    Run a single Monte Carlo Diffusion Limited Aggregation simulation.
    
    Parameters:
    -----------
    grid_size : int
        Size of the square grid
    max_growth_steps : int
        Maximum number of growth steps (particles added to cluster)
    iterations : int, optional
        Maximum number of iterations for animation (default: 1000)
    s_prob : float, optional
        Sticking probability (default: 0.5)
    show_animation : bool, optional
        Whether to show animation of the growth process (default: False)
    show_walkers : bool, optional
        Whether to display walkers in visualization (default: False)
    show_final : bool, optional
        Whether to display the final state (default: False)
        
    Returns:
    --------
    tuple
        (growth_steps, grid) - Number of growth steps completed and final grid state
    """
    # initialize the grid
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    grid[grid_size-1, grid_size//2] = 3

    # initialize the walkers and growth counter
    walkers = []
    growth_steps = 0
    reached_top = False

    if show_animation:
        fig, ax = plt.subplots()
        img = ax.imshow(grid, cmap='inferno', animated=True)

        def update(frame):
            nonlocal grid, walkers, growth_steps
            # Create a display grid that we'll use for visualization
            display_grid = grid.copy()
            
            # spawn a walker every iteration
            spawn_position, grid = spawn_walker(grid, grid_size)
            if spawn_position is not None:
                walkers.append(spawn_position)

            # update the position of each walker
            for j, walker in enumerate(walkers):
                new_position, grid, reached_top = update_walker_position(walker, grid, grid_size, s_prob)
                if new_position is None and walker is not None:  # A stick occurred
                    growth_steps += 1
                    if growth_steps >= max_growth_steps or reached_top:
                        plt.close()
                        return [img]
                if new_position is None:
                    walkers[j] = None
                else:
                    walkers[j] = new_position

            # Remove stuck walkers
            walkers = [walker for walker in walkers if walker is not None]
            
            # If we're not showing walkers, hide them in the display grid
            if not show_walkers:
                display_grid[grid == 2] = 0
            else:
                display_grid = grid.copy()
                
            img.set_array(display_grid)
            return [img]

        ani = animation.FuncAnimation(fig, update, frames=iterations, blit=True)
        plt.show()

    else:
        while growth_steps < max_growth_steps or reached_top:
            # spawn a walker every iteration
            spawn_position, grid = spawn_walker(grid, grid_size)
            if spawn_position is not None:
                walkers.append(spawn_position)

            # update the position of each walker
            for j, walker in enumerate(walkers):
                new_position, grid, reached_top = update_walker_position(walker, grid, grid_size, s_prob)
                if new_position is None and walker is not None:  # A stick occurred
                    growth_steps += 1
                    if growth_steps >= max_growth_steps or reached_top:
                        break
                if new_position is None:
                    walkers[j] = None
                else:
                    walkers[j] = new_position

            # Remove stuck walkers
            walkers = [walker for walker in walkers if walker is not None]

    # Prepare final visualization
    if not show_walkers:
        display_grid = grid.copy()
        display_grid[grid == 2] = 0
        grid = display_grid
    
    if show_final:
        plt.imshow(grid, cmap='inferno')
        plt.show()

    return growth_steps, grid

def compute_fractal_dimensions_mc_dla(grid):
    """
    Compute the fractal dimension of a Monte Carlo DLA cluster.
    
    Parameters:
    -----------
    grid : numpy.ndarray
        Grid containing the DLA cluster (cells with value 3)
        
    Returns:
    --------
    float
        The estimated fractal dimension of the cluster
    """
    sizes, counts = box_counting(grid)
    slope = fit_fractal_dimension(sizes, counts)
    return slope


if __name__ == "__main__":
    grid_size = 100
    iterations = 50000
    max_growth_steps = 1000
    probs = np.linspace(0.1, 1, 10)
    print('start')

    # Initialize fractal_dims as a dictionary of lists
    fractal_dims = {p: [] for p in probs}

    for i, sticking_prob in enumerate(probs):
        print(f'Running for sticking probability: {sticking_prob}')
        plt.subplot(1, len(probs), i + 1) 
        for _ in range(25):
            steps, grid = run_single_mc_dla(grid_size, max_growth_steps, iterations, sticking_prob, show_animation=False, show_walkers=False, show_final=False)
            # Append the fractal dimension to the list for this probability
            fractal_dims[sticking_prob].append(compute_fractal_dimensions_mc_dla(grid))
        plt.imshow(grid, cmap='inferno')
        plt.title(f'$p_s$: {sticking_prob}', fontsize=16)
        plt.axis('off')
        
    plt.suptitle('MC-DLA Growth Patterns for Different Sticking Probabilities', fontsize=18)
    plt.tight_layout()

    # Plot fractal dims mean and std
    plt.figure(figsize=(8, 5))
    means = [np.mean(fractal_dims[p]) for p in probs]
    stds = [np.std(fractal_dims[p]) for p in probs]

    # errorbars, 
    plt.errorbar(probs, means, yerr=stds, fmt='o', capsize=5, ecolor='grey', color='blue')
    
    plt.xlabel('Sticking Probability ($p_s$)')
    plt.ylabel('Fractal Dimension D')
    plt.title('Fractal Dimension vs. $p_s$')
    #dashed grid
    plt.grid(True, linestyle='--')
    plt.ylim(1.2, 1.8)
    plt.show()
