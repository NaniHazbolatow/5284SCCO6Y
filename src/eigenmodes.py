import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import eig
from scipy.sparse import diags, kron, identity, linalg, lil_matrix
from sympy import Matrix, latex
import time


def construct_M_square(N, h):
    """Constructs the matrix for the eigenvalue problem for a square membrane.

    Args:
        N (int): number of discretization steps
        h (float): spatial increment

    Returns:
        2D array: Matrix that encodes the 5-point stencil for the discrete laplacian.
    """    
    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N-1)
    sparse_mat = diags([main_diag, off_diag, off_diag], [0, -1, 1]) / h**2

    I = identity(N)
    M = kron(I, sparse_mat) + kron(sparse_mat, I)

    return M
    

def construct_M_rec(Nx, Ny, hx, hy):
    """Constructs the matrix for the eigenvalue problem for a rectangular membrane.

    Args:
        Nx (int): Number of discretization steps in the x direction
        Ny (int): Number of discretization steps in the y direction
        hx (float): spatial increment in the x direction
        hy (float): spatial increment in the y direction

    Returns:
        2D array: Matrix that encodes the 5-point stencil for the discrete laplacian.
    """    
    main_diag_x = -2 * np.ones(Nx)
    off_diag_x = np.ones(Nx-1)
    sparse_mat_x = diags([main_diag_x, off_diag_x, off_diag_x], [0, -1, 1]) / hx**2
    Ix = identity(Nx)

    main_diag_y = -2 * np.ones(Ny)
    off_diag_y = np.ones(Ny-1)
    sparse_mat_y = diags([main_diag_y, off_diag_y, off_diag_y], [0, -1, 1]) / hy**2
    Iy = identity(Ny)

    M = kron(Iy, sparse_mat_x) + kron(sparse_mat_y, Ix)

    return M
    

def construct_M_circle(N, h, L):
    """Constructs the matrix for the eigenvalue problem for a rectangular matrix.
       The Dirichlet boundary conditions are implemented by first creating a boolean matrix 
       that determines if a point is within the domain. Depending on the location of a grid
       point, a value is placed in.
    
    Args:
        N (int): Number of discretization steps
        h (float): spatial increment
        L (float): system size

    Returns:
        2D array: Matrix that encodes the 5-point stencil for the discrete laplacian.
    """    
    N = int(N)
    inside_circle = np.zeros((N,N), dtype=bool)
    centre = L / 2
    for i in range(N):
        for j in range(N):
            if np.sqrt((i*h - centre)**2 + (j*h - centre)**2) <= centre:
                inside_circle[i, j] = True

    def index_map(i, j): 
        return i * N + j

    M = lil_matrix((N*N, N*N))

    for i in range(N):
        for j in range(N):
            k = index_map(i, j)
            if inside_circle[i, j]:
                M[k, k] = -4
                # Above neighbor
                if i > 0 and inside_circle[i-1, j]:
                    M[k, index_map(i-1, j)] = 1
                # Below neighbor
                if i < N-1 and inside_circle[i+1, j]:
                    M[k, index_map(i+1, j)] = 1
                # Right neighbor
                if j < N-1 and inside_circle[i, j+1]:
                    M[k, index_map(i, j+1)] = 1
                # Left nieghbor
                if j > 0 and inside_circle[i, j-1]:
                    M[k, index_map(i, j-1)] = 1 
            else:
                M[k, k] = 1

    M = M / (h**2)

    return M


def plot_eigenvectors(lamda, v, grid_shape, L, plot_title, freqs_of_interest=[0, 1, 2, 3], shape='square'):
    """Sorts the eigenvalues/vectors from smallest to largest magnitude and plots the corresponding eigenmodes.

    Args:
        lamda (array): array of eigenvalues
        v (array): array of eigenvectors
        grid_shape (tuple): general shape of the grid. LxL for square and circle and Lx2L for rectangle.
        L (float): system size
        plot_title (str): title of the subplot, depends on the shape of the membrane
        freqs_of_interest (list, optional): list of modes that are plotted. Defaults to [0, 1, 2, 3].
        shape (str, optional): shape of the membrane. Defaults to 'square'.
    """    
    idx = np.flip(np.argsort(np.real(lamda)))
    lamda = lamda[idx]
    v = v[:, idx]

    freqs = np.sqrt(np.abs(np.real(lamda)))

    plt.figure(figsize=(5*len(freqs_of_interest), 5), dpi=400)
    plt.suptitle(f'{plot_title}', fontsize=21)
    for i, index in enumerate(freqs_of_interest):
        plt.subplot(1, len(freqs_of_interest), i + 1)
        if i == 0:
            plt.ylabel('y', fontsize=17)
        if shape == 'rectangle':
            Ny, Nx = grid_shape
            plt.imshow(np.real(v[:, index]).reshape(Ny,Nx), extent=[0,L,0,2*L], cmap='inferno')
            plt.yticks([0, 1, 2])
        else:
            N, _ = grid_shape
            plt.imshow(np.real(v[:, index]).reshape(N,N), extent=[0,L,0,L], cmap='inferno')
            plt.yticks([0, 1])
        plt.title(f'Frequency = {round(freqs[index], 2)}', fontsize=17)
        plt.xlabel('x', fontsize=17)
        plt.xticks([0, 1])
        plt.tick_params('both', labelsize=13)
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=13)
        cbar.set_label('Amplitude', fontsize=17)

    plt.tight_layout()
    plt.show()

def compute_execution_time(func, M, k=None):
    """Computes the execution time for calculating the eigenvalues/vectors for eig() and eigs()

    Args:
        func (function): solver: eig() or eigs()
        M (array): matrix of which eigenvalues/vectors are calculated
        k (int, optional): number of eigenvalues/vectors for eigs(). Defaults to None.

    Returns:
        float: execution time in nanoseconds
    """    
    start_eig = time.time_ns()
    if func == eig:
        lamda, v = func(M.toarray())
    else:
        lamda, v = func(M.toarray(), k)
    stop_eig = time.time_ns()

    return (stop_eig - start_eig)/1e6

def performance_compare(Ns, num_runs, plot=False):
    """Compares the performance between eigs() and eig() by measuring execution time over multiple runs for multiple system sizes.

    Args:
        Ns (array): array of different matrix sizes
        num_runs (int): number of runs per matrix size
        plot (bool, optional): option to plot results. Defaults to False.

    Returns:
        array: arrays containing the average and confidence intervals of the results.
    """    
    times_eig = np.zeros((num_runs, len(Ns)))
    times_eigs = np.zeros((num_runs, len(Ns)))
    times_eigs_long = np.zeros((num_runs, len(Ns)))

    for i in range(num_runs):
        for j, N in enumerate(Ns):

            M = construct_M_square(int(N), 1)
            times_eig[i, j] = compute_execution_time(eig, M)
            times_eigs[i,j] = compute_execution_time(linalg.eigs, M, k=6)
            times_eigs_long[i,j] = compute_execution_time(linalg.eigs, M, k=len(M.toarray()) - 2)

    mean_eig = np.mean(times_eig, axis=0)
    CI_eig = 1.96 * np.std(times_eig, axis=0) / np.sqrt(num_runs)
    mean_eigs = np.mean(times_eigs, axis=0)
    CI_eigs = 1.96 * np.std(times_eigs, axis=0) / np.sqrt(num_runs)
    mean_eigs_long = np.mean(times_eigs_long, axis=0)
    CI_eigs_long = 1.96 * np.std(times_eigs_long, axis=0) / np.sqrt(num_runs)

    if plot:
        plt.figure(figsize=(7, 5), dpi=400)
        plt.title('Performance Comparison of eig() and eigs()', fontsize=17)
        plt.scatter(Ns, mean_eig, color='black', label='eig() times')
        plt.fill_between(Ns, mean_eig - CI_eig, mean_eig + CI_eig, color='black', alpha=0.5)

        plt.scatter(Ns, mean_eigs, color='red', label='eigs(k=6) times')
        plt.fill_between(Ns, mean_eigs - CI_eigs, mean_eigs + CI_eigs, color='red', alpha=0.5)       

        plt.scatter(Ns, mean_eigs_long, color='orange', label='eigs(k=max) times')
        plt.fill_between(Ns, mean_eigs_long - CI_eigs_long, mean_eigs_long + CI_eigs_long, color='orange', alpha=0.5)  

        plt.xlabel('Grid Size', fontsize=15)
        plt.ylabel(r'Execution Time [ms]', fontsize=15)
        plt.tick_params('both', labelsize=12)
        plt.legend(fontsize=13)
        plt.yscale('log')

        plt.tight_layout()
        plt.show()  

    return mean_eig, CI_eig, mean_eigs, CI_eigs


def spectrum_vs_L(Ls, h, shape):
    """Calculates the eigenfrequencies as a function of system size for a given shape.

    Args:
        Ls (array): array of different system sizes
        h (float): spatial increment
        shape (str): shape of the membrane

    Returns:
        array: array of eigenfrequencies for the first 4 modes for different system sizes
    """    
    eigenfreqs = []
    for i, L in enumerate(Ls):
        if shape == 'rectangle':
            hx = h
            hy = 2*h
            Nx = int(L/hx) - 1
            Ny = int(2*L/hy) - 1
            M = construct_M_rec(Nx, Ny, hx, hy)

        elif shape == 'square':
            N = int(L/h) - 1
            M = construct_M_square(N, h)

        elif shape == 'circle':
            N = int(L/h) - 1
            M = construct_M_circle(N, h, L)
        else:
            raise ValueError('Choose a valid shape: square, rectangle, or circle.')

        lamda, _ = linalg.eigs(M, k=4, which='SM')
        
        idx = np.flip(np.argsort(np.real(lamda)))
        lamda = lamda[idx]

        freqs = np.sqrt(np.abs(np.real(lamda)))
        eigenfreqs.append(freqs)
    
    return eigenfreqs


def plot_spectrum_vs_L(Ls, eigenfreqs_sq, eigenfreqs_rec, eigenfreqs_cir):
    """Plots the results for the function spectrum_vs_L.

    Args:
        Ls (array): array of different system sizes
        eigenfreqs_sq (array): eigenfrequencies for different system sizes for a square membrane
        eigenfreqs_rec (array): eigenfrequencies for different system sizes for a rectangular membrane
        eigenfreqs_cir (array): eigenfrequencies for different system sizes for a circular membrane
    """    
    colors = ['gold', 'orange', 'red', 'black']
    plt.figure(figsize=(18, 5), dpi=400)
    plt.suptitle('Eigenfrequencies vs. System Size', fontsize=20)

    plt.subplot(1, 3, 1)
    plt.title('Square Membrane', fontsize=18)
    for mode in range(4):
        plt.plot(Ls, [freq[mode] for freq in eigenfreqs_sq], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel(r'System Size', fontsize=16)
    plt.ylabel('Eigenfrequency', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12)

    plt.subplot(1, 3, 2)
    plt.title('Rectangular Membrane', fontsize=18)
    for mode in range(4):
        plt.plot(Ls, [freq[mode] for freq in eigenfreqs_rec], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel(r'System Size', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12)

    plt.subplot(1, 3, 3)
    plt.title('Circular Membrane', fontsize=18)
    for mode in range(4):
        plt.plot(Ls, [freq[mode] for freq in eigenfreqs_cir], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel(r'System Size', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def spectrum_vs_num_steps(Ns, L, shape='square'):
    """Calculates the eigenfrequency spectrum for different numbers of discretization steps.

    Args:
        Ns (array): array with different numbers of discretization steps
        L (float): system size
        shape (str, optional): shape of the membrane. Defaults to 'square'.

    Returns:
        array: array of eigenfrequencies for the first 4 modes for different numbers of discretization steps
    """    
    freqs_vs_N = []
    for N in Ns:
        h = L/(int(N)+1)
        if shape == 'rectangle':
            M = construct_M_rec(int(N), int(N), h, 2*h)
        elif shape == 'square':
            M = construct_M_square(int(N), h)
        elif shape == 'circle':
            M = construct_M_circle(N, h, L)

        lamda, _ = linalg.eigs(M.tocsr(), k=4, which='SM')
        idx = np.flip(np.argsort(np.real(lamda)))
        lamda = lamda[idx]
        freqs = np.sqrt(np.abs(np.real(lamda)))
        freqs_vs_N.append(freqs)
    
    return freqs_vs_N


def plot_spectrum_vs_num_steps(Ns, freq_sq, freq_rec, freq_cir):
    """Plots the eigenfrequencies of the function spectrum_vs_num_steps.

    Args:
        Ns (array): array of different numbers of discretization steps
        eigenfreqs_sq (array): eigenfrequencies for different N for a square membrane
        eigenfreqs_rec (array): eigenfrequencies for different N for a rectangular membrane
        eigenfreqs_cir (array): eigenfrequencies for different N for a circular membrane
    """    
    colors = ['gold', 'orange', 'red', 'black']
    plt.figure(figsize=(18, 5), dpi=400)
    plt.suptitle('Eigenfrequencies vs. Number of Discretization Steps', fontsize=20)

    plt.subplot(1, 3, 1)
    plt.title('Square Membrane', fontsize=18)
    for mode in range(4):
        plt.plot(Ns, [freq[mode] for freq in freq_sq], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel('Number of Steps', fontsize=16)
    plt.ylabel('Eigenfrequency', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12)

    plt.subplot(1, 3, 2)
    plt.title('Rectangular Membrane', fontsize=18)
    for mode in range(4):
        plt.plot(Ns, [freq[mode] for freq in freq_rec], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel(r'Number of Steps', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12)

    plt.subplot(1, 3, 3)
    plt.title('Circular Membrane', fontsize=18)
    for mode in range(4):
        plt.plot(Ns, [freq[mode] for freq in freq_cir], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel(r'Number of Steps', fontsize=16)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()


def time_dependent_modes(time, c, lamda, v, N, mode_number=1, A=1, B=0):
    """Uses temporal component to make a time dependent animation of the eigenmodes.

    Args:
        time (array): array of time steps
        c (float): wave speed
        lamda (array): array of eigenfrequencies
        v (array): array of eigenmodes
        N (int): number of discretization steps
        mode_number (int, optional): mode to be animated. Defaults to 1, which is the first mode
        A (int, optional): multiplicative factor for the cosine. Defaults to 1.
        B (int, optional): multiplicative factor for the sine. Defaults to 0.

    Returns:
        matplotlib animation
    """    
    eigenmode_evo = []
    eigenmode = np.real(v[:, mode_number-1]).reshape(N,N)
    for t in time:
        u = eigenmode * (A * np.cos(c*np.real(lamda[mode_number-1])*t) +
                         B * np.sin(c*np.real(lamda[mode_number-1])*t))
        eigenmode_evo.append(u)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(eigenmode_evo[0], extent=[0, 1, 0, 1], cmap='inferno')
    ax.set_xlabel('x', fontsize=17)
    ax.set_ylabel('y', fontsize=17)
    ax.set_title('Animation of Eigenmodes', fontsize=16)
    cbar = fig.colorbar(im)
    cbar.set_label('Amplitude', fontsize=15)
    plt.tight_layout()

    def update(frame):
        im.set_array(eigenmode_evo[frame])
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(time), interval=50, blit=True)
    plt.close(fig)
    return ani


def draw_system_and_matrix(N, h, print_latex_matrix=False, plot_system=True):
    """Draw an example system and print out the latex code for it.

    Args:
        N (int): number of steps
        h (float): spatial increment
        print_latex_matrix (bool, optional): print latex code for matrix. Defaults to False.
        plot_system (bool, optional): plot example system. Defaults to True.
    """    
    M = construct_M_square(N, h)
    M_sympy = Matrix(M.toarray())
    if print_latex_matrix:
        print(latex(M_sympy))

    if plot_system:
        x = np.arange(N)
        y = np.arange(N)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(5, 5), dpi=300)
        plt.scatter(X, Y, color='black', s=50, zorder=10)
        plt.xticks(x)
        plt.yticks(y)
        plt.ylim(2.15, -0.15)
        plt.xlabel('x', fontsize=17)
        plt.ylabel('y', fontsize=17)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title(f"{N}x{N} Square Membrane", fontsize=19)
        plt.tight_layout()
        plt.show()
