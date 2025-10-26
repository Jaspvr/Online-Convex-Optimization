import cvxpy as cp

'''
We want to solve the optimization problem of projecting back onto the n-simplex
in the online newton step algorithm. What this actually becomes is a minimization of 
(x-y)A(x-y). Cncretely, we need to find the x on the simplex that minimizes (x-y)A(x-y).

Here, 
x: Vector of weights that lies on the simplex (sum to 1, each weight is >= 0)
y: Vector of weights after the newton step but before the projection
A: Matrix related to the hessian

Taking shapes into account we have (x-yt)^T At (x-yt)
'''

def cvxpyProjectToK(yt, At):
    n = yt.shape[0] # vector length

    # cvxpy variable: this is what cvxpy will optimize
    x = cp.Variable(n)

    # Quadratic objective: https://www.cvxpy.org/examples/basic/quadratic_program.html
    # Quadratic form is Q(z) = z^T A z
    objective = cp.Minimize(cp.quad_form(x - yt, At))

    # Simplex constraints
    constraints = [
        x >= 0,
        cp.sum(x) == 1
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    # Optimal x vector
    return x.value