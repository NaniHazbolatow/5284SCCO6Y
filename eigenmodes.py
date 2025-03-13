import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.sparse import diags, kron, identity, linalg, lil_matrix
import time
from sympy import Matrix, latex


def construct_M_square(N, h):

    main_diag = -2 * np.ones(N)
    off_diag = np.ones(N-1)
    sparse_mat = diags([main_diag, off_diag, off_diag], [0, -1, 1]) / h**2

    I = identity(N)
    M = kron(I, sparse_mat) + kron(sparse_mat, I)

    return M
    

def construct_M_rec(Nx, Ny, hx, hy):

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


def plot_eigenvectors(lamda, v, grid_shape, L, freqs_of_interest=[0, 1, 2, 3], shape='square'):

    idx = np.flip(np.argsort(np.real(lamda)))
    lamda = lamda[idx]
    v = v[:, idx]

    freqs = np.sqrt(np.abs(np.real(lamda)))

    plt.figure(figsize=(5*len(freqs_of_interest), 5), dpi=300)
    for i, index in enumerate(freqs_of_interest):
        plt.subplot(1, len(freqs_of_interest), i + 1)
        if shape == 'rectangle':
            Ny, Nx = grid_shape
            plt.imshow(np.real(v[:, index]).reshape(Ny,Nx), extent=[0,L,0,2*L], cmap='inferno')
        else:
            N, _ = grid_shape
            plt.imshow(np.real(v[:, index]).reshape(N,N), extent=[0,L,0,L], cmap='inferno')
        plt.title(f'Frequency = {round(freqs[index], 2)}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def performance_compare(Ns, num_runs, plot=False):

    times_eig = np.zeros((num_runs, len(Ns)))
    times_eigs = np.zeros((num_runs, len(Ns)))

    for i in range(num_runs):
        for j, N in enumerate(Ns):

            M = construct_M_square(int(N), 1)

            start_eig = time.time_ns()
            lamda, v = eig(M.toarray())
            stop_eig = time.time_ns()
            times_eig[i, j] = (stop_eig - start_eig)/1e6

            start_eigs = time.time_ns()
            lamda, v = linalg.eigs(M)
            stop_eigs = time.time_ns()
            times_eigs[i, j] = (stop_eigs - start_eigs)/1e6

    mean_eig = np.mean(times_eig, axis=0)
    CI_eig = 1.96 * np.std(times_eig, axis=0) / np.sqrt(num_runs)
    mean_eigs = np.mean(times_eigs, axis=0)
    CI_eigs = 1.96 * np.std(times_eigs, axis=0) / np.sqrt(num_runs)

    if plot:
        plt.figure(figsize=(7, 5), dpi=300)
        plt.title('Performance Comparison of eig() and eigs()', fontsize=17)
        plt.scatter(Ns, mean_eig, color='blue', label='eig() times')
        plt.fill_between(Ns, mean_eig - CI_eig, mean_eig + CI_eig, color='blue', alpha=0.5)

        plt.scatter(Ns, mean_eigs, color='red', label='eigs() times')
        plt.fill_between(Ns, mean_eigs - CI_eigs, mean_eigs + CI_eigs, color='red', alpha=0.5)       

        plt.xlabel(r'System Size $N$', fontsize=15)
        plt.ylabel(r'Execution Time [ms]', fontsize=15)
        plt.legend()

        plt.show()  

    return mean_eig, CI_eig, mean_eigs, CI_eigs


def spectrum_vs_L(Ls, h, shape):

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

        lamda, _ = linalg.eigs(M, k=4, which='SM')
        idx = np.flip(np.argsort(np.real(lamda)))
        lamda = lamda[idx]

        freqs = np.sqrt(np.abs(np.real(lamda)))
        eigenfreqs.append(freqs)
        
    colors = ['red', 'blue', 'green', 'black']
    plt.figure(figsize=(7, 5), dpi=300)
    plt.title('Eigenfrequencies vs. System Size', fontsize=17)
    for mode in range(4):
        plt.plot(Ls, [freq[mode] for freq in eigenfreqs], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel(r'System Size $L$', fontsize=15)
    plt.ylabel('Eigenfrequency', fontsize=15)
    plt.legend()
    plt.show()


def spectrum_vs_num_steps(Ns, L, shape='square'):

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

    colors = ['red', 'blue', 'green', 'black']
    plt.figure(figsize=(7,5), dpi=300)
    plt.title('Eigenfrequencies vs. Number of Steps', fontsize=17)
    for mode in range(4):
        plt.plot(Ns, [freq[mode] for freq in freqs_vs_N], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel('Number of Steps', fontsize=15)
    plt.ylabel('Eigenfrequency', fontsize=15)
    plt.legend()
    plt.show()


def time_dependent_modes(time, c, lamda, v, N, mode_number=1, A=1, B=0):
    
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
    plt.show()
    # plt.close(fig)
    return ani


def draw_system_and_matrix(N, h, print_latex_matrix=False, plot_system=True):
    M = construct_M_square(N, h)
    M_sympy = Matrix(M.toarray())
    if print_latex_matrix:
        print(latex(M_sympy))

    if plot_system:
        x = np.arange(4)
        y = np.arange(4)
        X, Y = np.meshgrid(x, y)

        plt.figure(figsize=(5, 5), dpi=300)
        plt.scatter(X, Y, color='black', s=50, zorder=10)
        plt.xticks(x)
        plt.yticks(y)
        plt.ylim(3.15, -0.15)
        plt.xlabel('x', fontsize=17)
        plt.ylabel('y', fontsize=17)
        plt.grid(linestyle='--', linewidth=0.5)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title("4x4 Square System", fontsize=17)
        plt.show()


# draw_system_and_matrix(4, 1, False, True)

######### MAKE ANIMATION OF EIGENMODES ###############
# c = 1
# L = 1
# h = 0.01
# N = int(L/h) - 1
# total_time = np.linspace(0, 20*np.pi, 100)
# M_square = construct_M_square(N, h)
# lamda, v = linalg.eigs(M_square, k=6, which='SM')
# idx = np.flip(np.argsort(np.real(lamda)))
# lamda = lamda[idx]
# v = v[:, idx]
# time_dependent_modes(total_time, c, lamda, v, N, mode_number=5)


########## PLOT SPECTRUM OF EIGENFREQUENCIES VS. SYSTEM SIZE ################
# Ls = np.linspace(0.25, 2, 10)
# h = 0.01
# spectrum_vs_L(Ls, h, 'square')
# spectrum_vs_L(Ls, h, 'rectangle')
# spectrum_vs_L(Ls, h, 'circle')




########## PLOT SPECTRUM OF EIGENFREQUENCIES VS. NUMBER OF DISCRETIZATION POINTS #############
# Ns = np.linspace(5, 50, 10)
# L = 1.0
# spectrum_vs_num_steps(Ns, L, 'square')
# spectrum_vs_num_steps(Ns, L, 'rectangle')
# spectrum_vs_num_steps(Ns, L, 'circle')


########## PLOT EIGENMODES AT DIFFERENT FREQUENCIES ############
# L = 1
# h = 0.01
# N = int(L/h) - 1
# M_square = construct_M_square(N, h)
# lamda, v = linalg.eigs(M_square, k=4, which='SM')
# plot_eigenvectors(lamda, v, grid_shape=(N,N), L=L)

# M_circle = construct_M_circle(N, h, L)
# lamda, v = linalg.eigs(M_circle, k=8, which='SM')
# plot_eigenvectors(lamda, v, grid_shape=(N,N), L=L, freqs_of_interest=[1, 3, 5, 7])

# L = 1
# hx = 0.01
# hy = 0.02
# Nx = int(L/hx) - 1
# Ny = int(2*L/hy) - 1
# M_rec = construct_M_rec(Nx, Ny, hx, hy)
# lamda, v = linalg.eigs(M_rec, k=4, which='SM')
# plot_eigenvectors(lamda, v, grid_shape=(Ny, Nx), L=L, shape='rectangle')




########## PLOT PERFORMANCE COMPARISON OF eig() vs. eigs() #############
# Ns = np.linspace(10, 20, 10)
# num_runs = 25
# mean_eig, CI_eig, mean_eigs, CI_eigs = performance_compare(Ns, num_runs)

# plt.figure(figsize=(7, 5), dpi=300)
# plt.title('Performance Comparison of eig() and eigs()', fontsize=17)
# plt.scatter(Ns, mean_eig, color='blue', label='eig() times')
# plt.fill_between(Ns, mean_eig - CI_eig, mean_eig + CI_eig, color='blue', alpha=0.5)

# plt.scatter(Ns, mean_eigs, color='red', label='eigs() times')
# plt.fill_between(Ns, mean_eigs - CI_eigs, mean_eigs + CI_eigs, color='red', alpha=0.5)       

# plt.xlabel(r'System Size $N$', fontsize=15)
# plt.ylabel(r'Execution Time [ms]', fontsize=15)
# plt.legend()

# plt.show()