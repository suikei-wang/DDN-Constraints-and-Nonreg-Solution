# DDN-Constraints-and-Nonreg-Solution
Some experiments of my research project on the Deep Declarative Network in 2020. I finally updated this repo and the main sections have been merged to the official public repo of DDN. 

### Part 1: Semester 1, 2020: Multiple constraints

Before the update, the source code of DDN in raw Python can only handle single equality or inequality constraint with the vanilla example. According to the gradient solution for multiple constraints problems and the code based on PyTorch developed by Dylan, I implemented the following `class` for both equality and inequality constraints problems associated with similar vanilla examples.  

#### Multiple Equality Constraints
For multiple equality constraints problems, `class` [MultiEqConstDeclarativeNode](https://github.com/suikei-wang/DDN-Constraints-and-Nonreg-Solution/blob/master/multiple_constraints/ddn/basic/node.py#L374) defines the constraints with optimality check and the gradient solution. It should be noticed that matrix H cannot be obtained through the inverse matrix directly. It should be solve through Cholesky factorization or linear programming. Size of the derivitives of the constraints array and objective function should be checked carefully for correct gradient dimension.   

#### Multiple Inequality Constraints
For multiple constraints problems containing both equality and inequality constraints, `class` [IneqConstDeclarativeNode](https://github.com/suikei-wang/DDN-Constraints-and-Nonreg-Solution/blob/master/multiple_constraints/ddn/basic/node.py#L193) defines functions for equality and inequality constraints with optimality check, and the gradient computation for both types of constraints. Different from the only equality constraints problems, the gradient is the combination of both equality and inequality constraints.


### Part 2: Semester 2, 2020: Non-regular solutions
In Part 1, we may find some error when we use some examples for both nodes. Sometimes, we get `nan` value when we calculate the `nu` in function `get_nu()`. This is caused by the non-regular solution in the contrained problems and we cannot get a the gradient of some constraints. 
It can be classified into [three following types](https://github.com/suikei-wang/DDN-Constraints-and-Nonreg-Solution/blob/master/multiple_constraints/07_non_regular_points.ipynb) and we give a possible solution for each of them:
- Overdetermined System: 
  - 1. **Least-Sequared Method**: Solving a least-square problem to minimize the residual r = Hx-b rathan than compute the matrix H directly.
  - 2. **Conjugate Gradient and Preconditioning**: Applying conjugate gradient method to do the gradient descent or approximating a matrix K closed to H using preconditioning.
- Rank Deficient (Underdetermined): 
  - 1. **Orthogonal Matching Pursuit Algorithm**: A signal recovery algorithm which recover the sparse signal vectors from a small number of noisy linear measurements.
- Non-convex Problem:
  - 1. **Non-linear Lagrangian**: Making the duality gap in Lagrange function always zero regardless of convexity and it is defined based on the weighted Chebyshev norm. 

