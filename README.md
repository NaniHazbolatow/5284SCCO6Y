# Scientific Computing at University of Amsterdam (5284SCCO6Y)
This is a repository for Scientific Computing (5284SCCO6Y) taken at the University of Amsterdam. This repository will contain all three assignments. 
However, for now, the current repository only contains the Python code required to recreate the answers and plots found in the report for the first assignment.

## Assignment 1
First, in order to be able to run the code without any errors, pull this repository and run `pip install -r requirements.txt` to install all Python packages used. 
All answers (and additional material, like animations) can be found by running [main.ipynb](https://github.com/NaniHazbolatow/5284SCCO6Y/blob/main/main.ipynb).
This Notebook imports custom functions from the following `.py` files:
*  [exercises1_1.py](src/exercise1_1.py): This `.py` files contains the functions used to answer sub-assignment 1.1, or questions A, B and C in the first assignment.
*  [exercises1_2.py](src/exercise1_2.py): This `.py` files contains the functions used to answer sub-assignment 1.2, or questions D, E, F and G in the first assignment.
*  [exercises1_6.py](src/exercise1_6.py): This `.py` files contains the functions used to answer sub-assignment 1.3, 1.4, 1.5, 1.6, or the remaining questions in the first assignment.
Throughout these files, `@njit` calls are made to parallelize some of the for-loops.


