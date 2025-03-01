import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def spawn_walker(grid, grid_size):
    available_positions = [y for y in range(grid_size) if grid[0, y] == 0]

    if not available_positions:
        
        #print("No available positions to spawn walker")
        return None, grid

    y_pos = available_positions[np.random.randint(0, len(available_positions))]
    spawn_position = [0, y_pos]
    grid[spawn_position[0], spawn_position[1]] = 2
    return spawn_position, grid
    
def update_walker_position(walker, grid, grid_size, s_prob):
    x, y = walker
    directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]

    # shuffle the directions
    np.random.shuffle(directions)
    for direction in directions:
        new_position = [x + direction[0], y + direction[1]]
        
        # check if horizontal wrap is needed
        if new_position[1] >= grid_size:
            new_position[1] = 0
        elif new_position[1] < 0:
            new_position[1] = grid_size-1
        
        # check if vertical elimination and respawn is needed
        if new_position[0] >= grid_size or new_position[0] < 0:
            grid[x, y] = 0
            spawn_position, grid = spawn_walker(grid, grid_size)
            return spawn_position, grid

        # check for sticking
        adjacent_positions = [[x + 1, y], [x - 1, y], [x, y + 1], [x, y - 1]]
        np.random.shuffle(adjacent_positions)
        for adj_pos in adjacent_positions:
            # if outside the grid, continue
            if adj_pos[0] >= grid_size or adj_pos[0] < 0 or adj_pos[1] >= grid_size or adj_pos[1] < 0:
                continue

            if grid[adj_pos[0], adj_pos[1]] == 1 and np.random.rand() < s_prob:
                grid[x, y] = 1
                return None, grid

        # check if the new position is valid and move the walker
        if grid[new_position[0], new_position[1]] == 0:
            grid[new_position[0], new_position[1]] = 2
            grid[x, y] = 0
            return new_position, grid

    return walker, grid

def run_single_mc_dla(grid_size, iterations=1000, s_prob=0.5):
    # initialize the grid
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    grid[grid_size-1, grid_size//2] = 1

    # initialize the walkers
    walkers = []

    for i in tqdm(range(iterations)):
        # spawn a walker every iteration
        spawn_position, grid = spawn_walker(grid, grid_size)
        if spawn_position is not None:
            walkers.append(spawn_position)

        # update the position of each walker
        for j, walker in enumerate(walkers):
            new_position, grid = update_walker_position(walker, grid, grid_size, s_prob)
            if new_position is None:
                walkers[j] = None
            else:
                walkers[j] = new_position

        # Remove stuck walkers
        walkers = [walker for walker in walkers if walker is not None]

    plt.imshow(grid, cmap='inferno')
    plt.show()


