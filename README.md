# Scientific Computing at University of Amsterdam (5284SCCO6Y)
This is a repository for Scientific Computing (5284SCCO6Y) taken at the University of Amsterdam. This repository will contain all three assignments. 
The code for the first assignment is stored in the folder named `Assignment_set_1`. The root directory contains the code for assignment set 2.

In order for the code to run smoothly, pull this repository and run `pip install -r requirements.txt` to install all Python packages used.

## Assignment 1
All answers (and additional material, like animations) can be found by running [main.ipynb](https://github.com/NaniHazbolatow/5284SCCO6Y/blob/main/main.ipynb).
This Notebook imports custom functions from the following `.py` files:
*  [exercises1_1.py](src/exercise1_1.py): This `.py` files contains the functions used to answer sub-assignment 1.1, or questions A, B and C in the first assignment.
*  [exercises1_2.py](src/exercise1_2.py): This `.py` files contains the functions used to answer sub-assignment 1.2, or questions D, E, F and G in the first assignment.
*  [exercises1_6.py](src/exercise1_6.py): This `.py` files contains the functions used to answer sub-assignment 1.3, 1.4, 1.5, 1.6, or the remaining questions in the first assignment.
Throughout these files, `@njit` calls are made to parallelize some of the for-loops.

## Assignment 2
The main python notebook contains all the plots and animations created using the code from the src folder.
*  [diffusion.py](src/diffusion.py): contains all the code to answer sub-assingment 2.1, which explores diffusion-limited aggregation.
*  [mc_dla.py](src/mc_dla.py): contains all the code to answer sub-assignment 2.2, which explores diffusion-limited aggregation with a Monte Carlo approach.
*  [gray_scott.py](src/gray_scott.py): contains all the code to answer sub-assignment 2.3, which explores the Gray-Scott model with finite difference methods.
