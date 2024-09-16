import pulp
import numpy as np
from scipy.optimize import linprog

def solve_linear_optimization(d, W, D):
    """
    Solves the linear optimization problem:
    min d.T @ x
    subject to W @ x = D
    and x >= 0

    Parameters:
    d (1D numpy array): Coefficients of the objective function.
    W (2D numpy array): Coefficient matrix for equality constraints.
    D (1D numpy array): Right-hand side vector for equality constraints.

    Returns:
    dict: A dictionary with variable values and the optimal objective function value.
    """
    
    # Ensure that the inputs are numpy arrays
    d = np.asarray(d)
    W = np.asarray(W)
    D = np.asarray(D)
    
    # Number of variables
    num_vars = d.shape[0]
    
    # Create a pulp linear problem object
    prob = pulp.LpProblem("Linear_Optimization", pulp.LpMinimize)
    
    # Create decision variables x, indexed from 0 to num_vars - 1, all non-negative
    x = pulp.LpVariable.dicts("x", range(num_vars), lowBound=0)
    
    # Objective function: minimize d.T @ x
    prob += pulp.lpSum(d[i] * x[i] for i in range(num_vars)), "Objective_Function"
    
    # Equality constraints: W @ x = D
    for i in range(W.shape[0]):
        
        prob += pulp.lpSum(W[i, j] * x[j] for j in range(num_vars)) == D[i], f"Constraint_{i}"
    
    # Solve the problem
    prob.solve()

      # Check if the problem was solved successfully
    if prob.status != pulp.LpStatusOptimal:
        if prob.status == pulp.LpStatusInfeasible:
            raise ValueError("The problem is infeasible.")
        elif prob.status == pulp.LpStatusUnbounded:
            raise ValueError("The problem is unbounded.")
        else:
            raise ValueError(f"Solver failed with status: {pulp.LpStatus[prob.status]}")
    
    # Extract the results
    result = {f"x_{i}": x[i].varValue for i in range(num_vars)}
    result['Objective_Value'] = pulp.value(prob.objective)
    
    return result


def get_basic_variable_indices(solution):
    """
    Returns the indices of basic variables (non-zero) in the solution.

    Parameters:
    solution (dict): The solution dictionary returned by the solve_linear_optimization function.

    Returns:
    list: A list of indices of the basic variables.
    """
    # Extract the indices of variables that are non-zero (basic variables)
    basic_indices = [int(var.split('_')[1]) for var, value in solution.items() if value != 0 and var != 'Objective_Value']
    
    return basic_indices


# def dual_optimal(d, W, D):
#     """
#     Computes the dual optimal solution.

#     Parameters:
#     d (1D numpy array): Coefficients of the primal objective function.
#     W (2D numpy array): Coefficient matrix for primal equality constraints.
#     solution (dict): Solution dictionary from the primal problem.

#     Returns:
#     numpy array: Dual optimal solution (Lagrange multipliers).
#     """

#     num_vars = W.shape[1]  # Number of decision variables (columns of W)
    
#     # Solve the primal LP using `linprog` (or another solver)
#     # We minimize d.T @ x subject to W @ x = D, and x >= 0
#     result = linprog(c=d, A_eq=W, b_eq=D, bounds=(0, None), method='highs')

#     if not result.success:
#         raise ValueError("LP solver failed to find a solution.")

#     # `result.slack` gives the slack for inequality constraints, 
#     # but we need the dual variables, which are provided in `result.pi`
    
#     # Extract dual values (Lagrange multipliers associated with equality constraints)
#     dual_solution = result.eqlin.marginals  # Dual variables for equality constraints

#     return dual_solution

# def dual_optimal(d, W, D):
#     """
#     Computes the dual optimal solution.

#     Parameters:
#     d (1D numpy array): Coefficients of the primal objective function.
#     W (2D numpy array): Coefficient matrix for primal equality constraints.
#     solution (dict): Solution dictionary from the primal problem.

#     Returns:
#     numpy array: Dual optimal solution (Lagrange multipliers).
#     """

#     solution = solve_linear_optimization(d, W, D)
#     basic_indices = get_basic_variable_indices(solution)
    
#     # Extract the matrix B corresponding to the basic variables (columns of W)
#     B = W[:, basic_indices]

#     # Extract the cost vector d_B corresponding to the basic variables
#     d_B = d[basic_indices]

#     # Handle the case where some elements in D are zero
#     dual_solution = np.zeros(W.shape[0])  # Dual solution for each constraint, initialized to zero
    
#     # Identify non-zero constraints
#     non_zero_indices = np.where(D != 0)[0]

#     if len(non_zero_indices) > 0:
#         # Matrix B for non-zero constraints
#         B_non_zero = W[non_zero_indices, :][:, basic_indices]
        
#         # Check if B is invertible for non-zero constraints
#         if np.linalg.matrix_rank(B_non_zero) < B_non_zero.shape[0]:
#             raise ValueError("Matrix B is singular; cannot compute the dual solution.")
        
#         # Compute the dual solution for non-zero constraints using the inverse of B
#         B_inv = np.linalg.inv(B_non_zero)
#         dual_solution_non_zero = np.dot(B_inv.T, d_B)
        
#         # Assign the computed dual solution to the corresponding non-zero constraint positions
#         dual_solution[non_zero_indices] = dual_solution_non_zero

#     # Dual solution for zero elements in D remains zero as initialized

#     return dual_solution

def dual_optimal(d,W,D):
    n, m = W.shape
    
    # Create the PuLP problem to maximize the objective function
    problem = pulp.LpProblem("Maximize_DTy", pulp.LpMaximize)
    
    # Define decision variables y as a list of continuous variables (unbounded)
    y = [pulp.LpVariable(f'y_{i}', lowBound=None, upBound=None) for i in range(n)]
    
    # Define the objective function: D^T * y
    # print(D[51])
    # print(y[51])
    problem += pulp.lpSum(D[i] * y[i] for i in range(n)), "Objective"
    W_y = W.T @ np.array(y)

    # Iterate over the columns (or constraints) and add them in one step
    constraints = [(W_y[j] <= d[j], f"Constraint_{j}") for j in range(m)]

    # Add all constraints at once
    for constraint in constraints:
        problem += constraint
    # Add the constraint W.T * y <= d
    # for j in range(m):
    #     problem += (pulp.lpSum(W[i, j] * y[i] for i in range(n)) <= d[j]), f"Constraint_{j}"
    
    # Solve the problem
    problem.solve(pulp.PULP_CBC_CMD(msg=False))  # Suppress the solver output
    # problem.solve()  # Suppress the solver output

    # Extract the solution as a numpy array
    y_solution = np.array([pulp.value(var) for var in y])
    
    return y_solution
def generate_random_arrays(N, M,seed):
    np.random.seed(seed)  # For reproducibility

    # Generate random d with shape (N,)
    d = np.random.rand(N)

    # Generate random W with shape (M, N)
    W = np.random.rand(M, N)

    # Generate random D with shape (M,)
    D = np.random.rand(M)

    # Force at least one zero in D
    zero_index = np.random.randint(0, M)
    D[zero_index] = 0

    return d, W, D
if __name__=="__main__":
    # print(np.random.randint(10))
    d,W,D=generate_random_arrays(1000,50,np.random.randint(1000))

    # print(d)
    # print(W)
    # print(D)
    # Compute the dual optimal solution
    dual_1 = dual_optimal(d, W, D)
    # dual_2 =dual_optimal_old(d,W,D)
    print(dual_1)
    # print("Dual Optimal Solution:", dual_1-dual_2)
# 