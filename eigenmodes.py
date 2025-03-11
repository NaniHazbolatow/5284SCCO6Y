import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity


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

M_square = construct_M(4, 1, 'square')



N = 10
h = 1.0 