import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def jacobi_iteration(N, eps, max_iter):
    """Solves 2D diffusion equation using a Jacobi iteration scheme.

    Args:
        N (int): grid size
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations

    Returns:
        The final grid after convergence is reached, an array of errors of each iteration, and the number of iterations it took to converge.
    """    
    grid = np.zeros((N, N))
    grid[0, :] = 1
    new_grid = grid.copy()
    deltas = np.zeros(max_iter)
    for k in range(max_iter):

        for i in range(1, N-1):
            for j in range(N):
                new_grid[i, j] = 1/4 * (grid[i+1, j] + grid[i-1, j] + grid[i, (j+1)%N] + grid[i, (j-1)%N])

        diff = np.max(np.abs(new_grid - grid))
        deltas[k] = diff
        if diff < eps:
            return new_grid, deltas[:k+1], np.arange(1, k+2)
        
        grid[:] = new_grid

    return new_grid, deltas[:k+1], np.arange(1, k+2)

@njit
def gauss_seidel_iteration(N, eps, max_iter):
    """Solves 2D diffusion equation using a Gauss-Seidel iteration scheme.

    Args:
        N (int): grid size
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations

    Returns:
        The final grid after convergence is reached, an array of errors of each iteration, and the number of iterations it took to converge.
    """
    grid = np.zeros((N, N))
    grid[0, :] = 1

    deltas = np.zeros(max_iter)
    for k in range(max_iter):
        diff = 0.0
        
        for i in range(1, N-1):
            for j in range(N):
                old_value = grid[i, j]
                grid[i, j] = (1/4) * (grid[i+1, j] + grid[i-1, j] + grid[i, (j+1)%N] + grid[i, (j-1)%N])
                
                diff = max(diff, abs(grid[i, j] - old_value))

        deltas[k] = diff
        if diff < eps:
            return grid, deltas[:k+1], np.arange(1, k+2)
            
    return grid, deltas[:k+1], np.arange(1, k+2)

@njit
def successive_over_relaxation(N, omega, eps, max_iter):
    """Solves 2D diffusion equation using a succesive over relaxation scheme.

    Args:
        N (int): grid size
        omega (float): relaxation parameter
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations

    Returns:
        The final grid after convergence is reached, an array of errors of each iteration, and the number of iterations it took to converge.
    """
    grid = np.zeros((N, N))
    grid[0, :] = 1

    deltas = np.zeros(max_iter)
    for k in range(max_iter):
        diff = 0.0
        
        for i in range(1, N-1):
            for j in range(N):
                prev_value = grid[i, j]
                new_value = omega/4 * (grid[i+1, j] + grid[i-1, j] + grid[i, (j+1)%N] + grid[i, (j-1)%N]) + (1 - omega)*prev_value

                grid[i, j] = new_value
                diff = max(diff, abs(new_value - prev_value))

        deltas[k] = diff
        if diff < eps:
            return grid, deltas[:k+1], np.arange(1, k+2)
        
    return grid, deltas, np.arange(1, max_iter+1)

def compare_methods_to_analytical(N, omega, eps, max_iter):
    """Compares and plots the final concentration along the y-axis for different methods: Jacobi, Gauss-Seidel, and SOR.

    Args:
        N (int): grid size
        omega (float): relaxation parameter
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations
    """    
    grid_jacobi, _, _ = jacobi_iteration(N, eps, max_iter)
    grid_gauss, _, _ = gauss_seidel_iteration(N, eps, max_iter)
    grid_SOR, _, _ = successive_over_relaxation(N, omega, eps, max_iter)

    y = np.linspace(0, 1, N)
    c_analytical = y
    c_jacobi = np.flip(grid_jacobi[:, 0])
    c_gauss = np.flip(grid_gauss[:, 0])
    c_SOR = np.flip(grid_SOR[:, 0])

    plt.figure(figsize=(7, 5), dpi=300)
    plt.title('Concentration at Late Times for Different Methods', fontsize=15)
    plt.plot(y, c_analytical, color='black', ls='dotted', label='Analytical', zorder=2.5)
    plt.plot(y, c_jacobi, color='blue', label='Jacobi Iteration')
    plt.plot(y, c_gauss, color='green', label='Gauss-Seidel Iteration')
    plt.plot(y, c_SOR, color='red', label='SOR')
    plt.xlabel('y', fontsize=14)
    plt.ylabel('Concentration', fontsize=14)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()

def convergence_measure(N, omegas, eps, max_iter):
    """Calculates and plots the deltas versus the number of iterations for different methods: Jacobi, Gauss-Seidel, and SOR with varying omega.

    Args:
        N (int): grid size
        omega (float): relaxation parameter
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations
    """    
    _, deltas_jacobi, iter_jacobi = jacobi_iteration(N, eps, max_iter)
    _, deltas_gauss, iter_gauss = gauss_seidel_iteration(N, eps, max_iter)
    _, deltas_SOR_75, iter_SOR_75 = successive_over_relaxation(N, omegas[0], eps, max_iter)
    _, deltas_SOR_85, iter_SOR_85 = successive_over_relaxation(N, omegas[1], eps, max_iter)
    _, deltas_SOR_95, iter_SOR_95 = successive_over_relaxation(N, omegas[2], eps, max_iter)

    print(f'SOR (omega=1.95): {iter_SOR_95[-1]}\nGauss: {iter_gauss[-1]}\nJacobi: {iter_jacobi[-1]}')

    plt.figure(figsize=(7, 5), dpi=300)
    plt.title('Convergence for Different Numerical Schemes', fontsize=15)
    plt.loglog(iter_SOR_75, deltas_SOR_75, color='gold', label=fr'SOR, $\omega = {{{omegas[0]}}}$')
    plt.loglog(iter_SOR_85, deltas_SOR_85, color='orange', label=fr'SOR, $\omega = {{{omegas[1]}}}$')
    plt.loglog(iter_SOR_95, deltas_SOR_95, color='red', label=fr'SOR, $\omega = {{{omegas[2]}}}$')
    plt.loglog(iter_gauss, deltas_gauss, color='green', label='Gauss-Seidel Iteration')
    plt.loglog(iter_jacobi, deltas_jacobi, color='blue', label='Jacobi Iteration')
    plt.xlabel('Number of Iterations', fontsize=15)
    plt.ylabel(r'$\delta$', fontsize=15)
    plt.legend()
    plt.tick_params(axis='both', labelsize=12)
    plt.tight_layout()
    plt.show()

def optimal_omega(Ns, omegas, eps, max_iter, plot=True):
    """Calculates and plots the optimal omega for different grid sizes.

    Args:
        Ns (array): array of different grid sizes
        omegas (array): array of omega values
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations
        plot (bool, optional): option to plot omega vs. N. Defaults to True.

    Returns:
        list: optimal values of omega for different N
    """    
    optimal_omegas = []
    for j, N in enumerate(Ns):

        iters = np.zeros_like(omegas)
        for i, omega in enumerate(omegas):
            _, _, iter_SOR = successive_over_relaxation(int(N), omega, eps, max_iter)
            iters[i] = iter_SOR[-1]

        optimal_omegas.append(omegas[np.argmin(iters)])

    if plot:
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title('Optimal Omega for SOR method vs. Grid Size', fontsize=15)
        plt.scatter(Ns, optimal_omegas, color='blue')
        plt.xlabel('N', fontsize=15)
        plt.ylabel(r'Optimal $\omega$', fontsize=15)
        plt.tick_params(axis='both', labelsize=12)
        plt.tight_layout()
        plt.show()

    return optimal_omegas


def init_objects(objects, N):
    """Initialises some object into a grid

    Args:
        objects (array): array of indeces corresponding to the bounds of the object: i_min, i_max, j_min, j_max.
        N (int): grid size

    Returns:
        array: grid with all objects
    """    
    grid = np.zeros((N, N))
    grid[0, :] = 1
    for obj in objects:
        i_min, i_max, j_min, j_max = obj
        if i_min > i_max or j_min > j_max:
            raise ValueError('i_min and j_min have to be smaller than i_max and j_max')

        grid[int(i_min):int(i_max+1), int(j_min):int(j_max+1)] = 1
    grid[-1, :] = 0
    return grid

@njit
def SOR_object(object_grid, N, omega, eps, max_iter):
    """Solves 2D diffusion equation using a successive over relaxation scheme.

    Args:
        object_grid (NDarray): grid with objects as 1s
        N (int): grid size
        omega (float): relaxation parameter
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations

    Returns:
        The final grid after convergence is reached, an array of errors of each iteration, and the number of iterations it took to converge.
    """    
    grid = np.zeros((N, N))
    grid[0, :] = 1
    deltas = np.zeros(max_iter)
    for k in range(max_iter):
        diff = 0.0
        
        for i in range(1, N-1):
            for j in range(N):
                if object_grid[i, j] == 1:
                    grid[i, j] = 0
                else:
                    prev_value = grid[i, j]
                    new_value = omega/4 * (grid[i+1, j] + grid[i-1, j] + grid[i, (j+1)%N] + grid[i, (j-1)%N]) + (1 - omega)*prev_value

                    grid[i, j] = new_value
                    diff = max(diff, abs(new_value - prev_value))

        deltas[k] = diff
        if diff < eps:
            return grid, deltas[:k+1], np.arange(1, k+2)
        
    return grid, deltas, np.arange(1, max_iter+1)


def convergence_with_objects(objects, N, omega, eps, max_iter):
    """Calculates and plots the convergence vs. the number of iterations for the SOR method with a different number of concentration sinks.

    Args:
        object_grid (NDarray): grid with objects as 1s
        N (int): grid size
        omega (float): relaxation parameter
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations
    """    
    grid_with_object_1 = init_objects(objects[0], N)
    _, deltas_obj_1, iter_obj_1 = SOR_object(grid_with_object_1, N, omega, eps, max_iter)

    grid_with_object_2 = init_objects(objects[1], N)
    _, deltas_obj_2, iter_obj_2 = SOR_object(grid_with_object_2, N, omega, eps, max_iter)

    grid_with_object_3 = init_objects(objects[2], N)
    _, deltas_obj_3, iter_obj_3 = SOR_object(grid_with_object_3, N, omega, eps, max_iter)

    # Without objects
    _, deltas, iter = successive_over_relaxation(N, omega, eps, max_iter)

    print(f'SOR (0 sinks): {iter[-1]}\nSOR (1 sink): {iter_obj_1[-1]}\nSOR (2 sink): {iter_obj_2[-1]}\nSOR (3 sink): {iter_obj_3[-1]}')


    plt.figure(figsize=(7, 5), dpi=300)
    plt.title('Convergence vs. Iterations with Objects', fontsize=15)
    plt.loglog(iter_obj_1, deltas_obj_1, color='red', label='SOR with 1 square')
    plt.loglog(iter_obj_2, deltas_obj_2, color='blue', label='SOR with 2 squares')
    plt.loglog(iter_obj_3, deltas_obj_3, color='green', label='SOR with 3 squares')
    plt.loglog(iter, deltas, color='black', label='SOR without objects')
    plt.xlabel('Number of Iterations', fontsize=15)
    plt.ylabel(r'$\delta$', fontsize=15)
    plt.tick_params(axis='both', labelsize=12)
    plt.legend()
    plt.tight_layout()
    plt.show()


def optimal_omega_with_objects(objects, N, omegas, eps, max_iter):
    """Calculates the optimal value for omega when concentration sinks are introduced.

    Args:
        object_grid (NDarray): grid with objects as 1s
        N (int): grid size
        omega (float): relaxation parameter
        eps (float): convergence criteria
        max_iter (int): maximum number of iterations
    """    
    optimal_omegas = []

    for j, obj in enumerate(objects):

        iters = np.zeros_like(omegas)
        for i, omega in enumerate(omegas):
            grid_with_obj = init_objects(obj, N)
            _, _, iter_SOR = SOR_object(grid_with_obj, N, omega, eps, max_iter)
            iters[i] = iter_SOR[-1]
    
        optimal_omegas.append(omegas[np.argmin(iters)])
    
        print(f'The optimal omega with {j+1} square(s) is {optimal_omegas[j]}')
