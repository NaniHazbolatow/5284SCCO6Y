# Scientific Computing at University of Amsterdam (5284SCCO6Y)
This is a repository for Scientific Computing (5284SCCO6Y) taken at the University of Amsterdam. This repository will contain all three assignments. 
The code for the first and second parts are stored in the folders named `Assignment_set_1` and `Assignment_set_2`, respectively. The root directory contains the code for simulating vibrating membranes, the solution to a steady-state diffusion problem, and time integration with the leapfrog method.

In order for the code to run smoothly, pull this repository and run `pip install -r requirements.txt` to install all Python packages used.

## Finite difference methods for the wave equation and diffusion
All answers (and additional material, like animations) can be found in the folder `Assignment_set_1`.
This Notebook imports custom functions from the following `.py` files:
*  [exercises1_1.py](src/exercise1_1.py): This `.py` files contains the functions used to answer sub-assignment 1.1, or questions A, B and C in the first assignment.
*  [exercises1_2.py](src/exercise1_2.py): This `.py` files contains the functions used to answer sub-assignment 1.2, or questions D, E, F and G in the first assignment.
*  [exercises1_6.py](src/exercise1_6.py): This `.py` files contains the functions used to answer sub-assignment 1.3, 1.4, 1.5, 1.6, or the remaining questions in the first assignment.
Throughout these files, `@njit` calls are made to parallelize some of the for-loops.

## Diffusion-Limited Aggregation and the Gray-Scott model
All answers (and additional material, like animations) can be found in the folder `Assignment_set_2`.
The main python notebook contains all the plots and animations created using the code from the src folder.
*  [diffusion.py](src/diffusion.py): contains all the code for diffusion-limited aggregation (DLA), solved using succesive over relaxation.
*  [mc_dla.py](src/mc_dla.py): contains all the code for DLA with a Monte Carlo approach.
*  [gray_scott.py](src/gray_scott.py): contains all the code to simulate the Gray-Scott model with finite difference methods.
The animations of the gray-scott model can be found in the last four cells of the notebook. The user may vary the parameter combinations to obtain different patterns. An example of most patterns is listed in (J. E. Pearson, Science, Vol 261, 5118, 189-192 (1993)). 

## Eigenmodes, Steady-state Systems, and efficient time integration
The main python notebook contains all the plots and animations created using the code from the src folder.
*  [eigenmodes.py](src/eigenmodes.py): contains all the code for the generation of the sparse matrices, and the plots of the membranes.
*  [steadystate.py](src/steadystate.py): contains all the code for the direct solver technique used to solve and plot the diffusive steady state in a circular domain.
*  [leapfrog.py](src/leapfrog.py): contains all the code for the efficient time integration using the leapfrog method.
The animations of the vibrating membranes can be found in the folder `Animations_Membrane`.
