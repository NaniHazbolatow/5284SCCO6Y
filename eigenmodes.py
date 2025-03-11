import numpy as np
from scipy.linalg import eig
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity, linalg
import time


def construct_M(N, h, shape='square'):

    if shape == 'square':
        main_diag = -2 * np.ones(N)
        off_diag = np.ones(N-1)
        sparse_mat = diags([main_diag, off_diag, off_diag], [0, -1, 1]) / h**2

        I = identity(N)
        M = kron(I, sparse_mat) + kron(sparse_mat, I)

        return M
    
    elif shape == 'rectangle':
        return
    
    elif shape == 'circle':
        main_diag = -2 * np.ones(N)
        off_diag = np.ones(N-1)
        sparse_mat = diags([main_diag, off_diag, off_diag], [0, -1, 1]) / h**2

        I = identity(N)
        M = kron(I, sparse_mat) + kron(sparse_mat, I)

        centre = N/2 - 1
        for i in range(N):
            for j in range(N):
                d = np.sqrt((i-centre)**2 + (j-centre)**2)
                if d > N/2:
                    M[i, j] = 0

        return M
    
    else:
        raise ValueError('Not a valid shape. Use \'square\', \'rectangle\', or \'circle\'.')

def plot_eigenvectors(lamda, v):

    idx = np.flip(np.argsort(np.real(lamda)))
    lamda = lamda[idx]
    v = v[:, idx]

    freqs = np.sqrt(np.abs(np.real(lamda)))

    plt.figure(figsize=(15, 5))
    freqs_of_interest = [0, 1, 3]
    for i, index in enumerate(freqs_of_interest):
        plt.subplot(1, len(freqs_of_interest), i + 1)
        plt.imshow(np.real(v[:, index]).reshape(N,N), extent=[0,1,0,1], cmap='inferno')
        plt.title(f'Frequency = {round(freqs[index], 2)}')
        plt.colorbar()
    
    plt.tight_layout()
    plt.show()


def performance_compare(Ns, num_runs):

    times_eig = np.zeros((num_runs, len(Ns)))
    times_eigs = np.zeros((num_runs, len(Ns)))

    for i in range(num_runs):
        for j, N in enumerate(Ns):

            M = construct_M(int(N), 1, 'square')

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

    return mean_eig, CI_eig, mean_eigs, CI_eigs


def spectrum_vs_L(Ls, h, shape):

    eigenfreqs = []
    for i, L in enumerate(Ls):
        N = int(L/h) - 1
        M = construct_M(N, h, shape)
        lamda, _ = linalg.eigs(M, k=4, which='SM')

        idx = np.flip(np.argsort(np.real(lamda)))
        lamda = lamda[idx]

        freqs = np.sqrt(np.abs(np.real(lamda)))
        eigenfreqs.append(freqs)
        
    colors = ['red', 'blue', 'green', 'black']
    plt.figure(figsize=(7, 5))
    plt.title('Eigenfrequencies vs. System Size', fontsize=17)
    for mode in range(4):
        plt.plot(Ls, [freq[mode] for freq in eigenfreqs], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel(r'System Size $L$', fontsize=15)
    plt.ylabel('Eigenfrequency', fontsize=15)
    plt.legend()
    plt.show()

def spectrum_vs_num_steps(Ns, L):

    freqs_vs_N = []
    for N in Ns:
        h = L/(int(N)+1)
        M = construct_M(int(N), h, 'square')
        lamda, _ = linalg.eigs(M.tocsr(), k=4, which='SM')
        idx = np.flip(np.argsort(np.real(lamda)))
        lamda = lamda[idx]
        freqs = np.sqrt(np.abs(np.real(lamda)))
        freqs_vs_N.append(freqs)

    colors = ['red', 'blue', 'green', 'black']
    plt.figure(figsize=(7,5))
    plt.title('Eigenfrequencies vs. Number of Steps', fontsize=17)
    for mode in range(4):
        plt.plot(Ns, [freq[mode] for freq in freqs_vs_N], marker='o', label=f'Mode {mode+1}', color=colors[mode])
    plt.xlabel('Number of Steps', fontsize=15)
    plt.ylabel('Eigenfrequency', fontsize=15)
    plt.legend()
    plt.show()

# Ls = np.linspace(0.5, 2, 10)
# h = 0.01
# spectrum_vs_L(Ls, h, 'square')

Ns = np.linspace(5, 100, 20)
L = 1.0
spectrum_vs_num_steps(Ns, L)


# L = 1
# h = 0.01
# N = int(L/h) - 1
# M_square = construct_M(N, h, 'square')
# lamda, v = linalg.eigs(M_square, k=4, which='SM')


# plot_eigenvectors(lamda, v)




# Ns = np.linspace(10, 20, 10)
# num_runs = 25
# mean_eig, CI_eig, mean_eigs, CI_eigs = performance_compare(Ns, num_runs)

# plt.figure(figsize=(7, 5))
# plt.title('Performance Comparison of eig() and eigs()', fontsize=17)
# plt.scatter(Ns, mean_eig, color='blue', label='eig() times')
# plt.fill_between(Ns, mean_eig - CI_eig, mean_eig + CI_eig, color='blue', alpha=0.5)

# plt.scatter(Ns, mean_eigs, color='red', label='eigs() times')
# plt.fill_between(Ns, mean_eigs - CI_eigs, mean_eigs + CI_eigs, color='red', alpha=0.5)       

# plt.xlabel(r'System Size $N$', fontsize=15)
# plt.ylabel(r'Execution Time [ms]', fontsize=15)
# plt.legend()

# plt.show()