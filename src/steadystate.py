import numpy as np
import scipy.sparse.linalg
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.sparse import diags, kron, identity, csr_matrix
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def Laplace1D(N: int):
    """
    Construct the 1D Laplace operator as a sparse tridiagonal matrix.

    The operator approximates the second derivative on a 1D grid with N points 
    using finite differences.

    Args:
        N (int): Number of grid points.

    Returns:
        csr_matrix: A sparse matrix representing the 1D Laplacian operator.
    """
    e = np.ones(N)
    lap1D = diags([e, -2*e, e], offsets=[-1, 0, 1], shape=(N, N), format='csr')
    return lap1D

def Laplace2D(N: int):
    """
    Construct the 2D Laplace operator for a square grid using Kronecker products.

    The operator is built by combining 1D Laplace operators for the x and y directions.

    Args:
        N (int): Number of grid points in one dimension.

    Returns:
        csr_matrix: A sparse matrix representing the 2D Laplacian operator.
    """
    lap1D = Laplace1D(N)
    lap2D = kron(identity(N, format='csr'), lap1D) + kron(lap1D, identity(N, format='csr'))
    return lap2D

class DiffusiveCircle:
    """
    Class for simulating diffusion in a circular domain.

    The domain is defined as a circle of radius 'r' and a grid with a specified step size.
    Diffusion is simulated by solving the steady-state Laplace equation with Dirichlet
    boundary conditions imposed at specified sink points.
    """
    def __init__(self, r: float, step=0.01):
        """
        Initialize the DiffusiveCircle instance.

        Args:
            r (float): The radius of the circular domain.
            step (float, optional): The grid spacing used to discretize the domain. 
                                    Defaults to 0.01.
        """
        self.radius = r
        self.step_size = step

        # Create a grid covering [-r, r] in both x and y.
        self.X, self.Y = np.meshgrid(
            np.arange(-self.radius, self.radius + self.step_size, self.step_size),
            np.arange(-self.radius, self.radius + self.step_size, self.step_size),
        )

        # Identify points inside the circle: x^2 + y^2 <= r^2.
        self.interior_circle = self.X**2 + self.Y**2 <= self.radius**2

    def initialize_concentration(self, sinks: tuple):
        """Construct the initial concentration vector.
        
        Args:
            sinks (tuple): A tuple of coordinates, (x,y), indicating where the concentration should be 1.
        """        

        # Create concentration matrix with default value 0.
        self.concentration = np.zeros_like(self.X)

        # Compute grid indices for a given sink (x, y)
        def get_sink_indices(sink):
            return (
                int((sink[1] + self.radius) / self.step_size),
                int((sink[0] + self.radius) / self.step_size),
            )

        # Apply the Dirichlet condition (c = 1) for all sinks.
        for sink in sinks:
            i, j = get_sink_indices(sink)
            self.concentration[i, j] = 1.0

    def build_coefficient_matrix(self):
        """
        Build the coefficient matrix for the diffusion problem.

        This involves:
            - Constructing the 2D Laplace operator.
            - Masking the operator to the interior of the circle.
            - Applying Dirichlet boundary conditions by modifying rows corresponding to 
              points outside the circle or at sink points.

        Returns:
            csr_matrix: The modified sparse coefficient matrix for the linear system.
        """
        # Initialize 2D Laplace operator.
        M = Laplace2D(self.concentration.shape[0])
    
        # Create mask: 1 for points inside the circle, 0 outside.
        P = self.interior_circle.flatten().astype(float)
        D = diags(P, format='csr')
    
        # Apply the mask to the Laplacian.
        MP = D @ M @ D

        # Convert to LIL format for efficient row modifications.
        MP = MP.tolil()

        # Create a single boolean mask for indices that must enforce a Dirichlet condition:
        # either outside the circle (P == 0) or where concentration is fixed (sink: c == 1).
        fix_mask = (P == 0) | (np.isclose(self.concentration.flatten(), 1.0))
        fix_indices = np.where(fix_mask)[0]

        # In one pass, loop over all indices that need to be fixed.
        for idx in fix_indices:
            MP.rows[idx] = [idx]  # Keep only the diagonal entry.
            MP.data[idx] = [1.0]  # Set the diagonal to 1.

        # Convert back to CSR for efficient arithmetic and solving.
        return MP.tocsr()

    def solve_steady_state(self):
        """
        Solve for the steady state concentration distribution.

        This method builds the coefficient matrix, formulates the right-hand side vector, 
        and solves the resulting sparse linear system. The solution is reshaped to match 
        the grid dimensions and stored in the concentration attribute.
        """
        M = self.build_coefficient_matrix()

        # Right-hand side vector (0 for non-sinks, 1 for sinks).
        b = self.concentration.flatten()

        # Solve the sparse linear system.
        u = scipy.sparse.linalg.spsolve(M, b)

        # Reshape the solution to match the grid dimensions.
        self.concentration = u.reshape(self.concentration.shape)

    def plot_concentration(self, norm = mcolors.Normalize(vmin=0, vmax=1), zoom=None, zoom_factor=2, zoom_loc='lower right'):
        """
        Plot the steady-state concentration distribution in the circular domain.

        The function displays the concentration as a heatmap and overlays the circular domain.
        An optional zoom parameter can be provided to focus on a specific region.

        Args:
            norm (Normalize, optional): A matplotlib.colors.Normalize instance for scaling color values in zoom.
                                          Defaults to a normalization from 0 to 0.1.
            zoom (tuple, optional): A tuple of two tuples ((xmin, xmax), (ymin, ymax)) to specify the 
                                    zoomed region. Defaults to None.
        """
        plt.figure(dpi=150)
        extent = [-self.radius, self.radius, -self.radius, self.radius]
        plt.imshow(self.concentration, origin='lower', extent=extent,
               cmap='inferno')
        plt.colorbar(label='Concentration')
        plt.title('Diffusive Concentration in a Circle', fontsize=14)
        plt.xlabel('X', fontsize=12)
        plt.ylabel('Y', fontsize=12)

        # Overlay the circle domain.
        ax = plt.gca()
        circle_patch = Circle((0, 0), self.radius, facecolor='gray', alpha=0.15, edgecolor='gray', lw=2)
        ax.add_patch(circle_patch)

        if zoom is not None:
            # Create the inset axis.
            axins = zoomed_inset_axes(ax, zoom=zoom_factor, loc=zoom_loc)
            
            # Plot the same image on the inset.
            axins.imshow(self.concentration, origin='lower', extent=extent, cmap='inferno', norm=norm)
            # Set the limits for the zoomed region.
            axins.set_xlim(zoom[0])
            axins.set_ylim(zoom[1])
            # Hide tick labels on the inset.
            axins.set_xticks([])
            axins.set_yticks([])

            axins.add_patch(Circle((0, 0), self.radius, facecolor='gray', alpha=0.15, edgecolor='gray', lw=2))
            # Draw a box on the main plot to show the zoom region.
            mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])

        #plt.tight_layout()
        plt.show()

