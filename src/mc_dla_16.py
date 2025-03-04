import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from diffusion import box_counting, fit_fractal_dimension

def spawn_walker(grid, grid_size):
    available_positions = [y for y in range(grid_size) if grid[0, y] == 0]

    if not available_positions:
        return None, grid

    y_pos = available_positions[np.random.randint(0, len(available_positions))]
    spawn_position = [0, y_pos]
    grid[spawn_position[0], spawn_position[1]] = 2
    return spawn_position, grid
    
def update_walker_position(walker, grid, grid_size, s_prob):
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

def compute_fractal_dimension(grid):
    sizes, counts = box_counting(grid)
    slope = fit_fractal_dimension(sizes, counts)
    return slope


if __name__ == "__main__":
    grid_size = 50
    iterations = 50000
    max_growth_steps = 10 
    steps, grid = run_single_mc_dla(grid_size, max_growth_steps, iterations, 0.5, show_animation=False, show_walkers=False, show_final=True)
    slope = compute_fractal_dimension(grid)
    print(f"Number of growth steps: {steps}")
    print(f'Fractal dimension: {slope}')
    