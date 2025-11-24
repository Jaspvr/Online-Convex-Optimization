import cvxpy as cp
import numpy as np

def cvxpyOnsProjectToK(yt, At):
    '''
    ONS CVXPY projection.
    We want to solve the optimization problem of projecting back onto the n-simplex
    in the online newton step algorithm. What this actually becomes is a minimization of 
    (x-y)A(x-y). Cncretely, we need to find the x on the simplex that minimizes (x-y)A(x-y).

    Here, 
    x: Vector of weights that lies on the simplex (sum to 1, each weight is >= 0)
    y: Vector of weights after the newton step but before the projection
    A: Matrix related to the hessian

    Taking shapes into account we have (x-yt)^T At (x-yt)
    '''
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


def cvxpyOgdProjectToK(yt):
    '''
    Same as ONS except simplified since At is I
    '''
    n = yt.shape[0] # vector length

    # cvxpy variable: this is what cvxpy will optimize
    x = cp.Variable(n)

    # Euclidean distance: ||x - y||_2 ^ 2
    objective = cp.Minimize(cp.sum_squares(x-yt))

    # Simplex constraints
    constraints = [
        x >= 0,
        cp.sum(x) == 1
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    return x.value


def simpleONSProjectToK(self, yt, At):
        ''' This function projects onto the simplex (sum of weights = 1, each weight >= 0).
        ONS accomplishes this by finding the closest feasible portfolio to y in the At induced
        norm ('At' takes into account 2nd order information). Essentially we project back 
        using curvature-aware distance.
        '''

        # To get this projection, we need to find x on the simplex that minimizes
        # transpose(x-y)*A*(x-y). The gradient of this wrt x is A(x-y). What we can
        # do is gradient descent on this minimization by using repeated euclidean
        # projections while reducing x by its gradient each time to approach the solution

        alpha = 1e-3 # This should be improved upon/researched further
        xt = projectToK(yt)
        for _ in range(50):
            gt = At @ (xt - yt)
            xt = projectToK(xt - alpha * gt)

        return xt


def projectToK(y):
        ''' K is defined as an n-dimensional simplex with the rule of x >= 0 and sum of x = 1. 
        This function takes a weight decision vector and projects it to land in K if it does not already.
        A Euclidean projection can be used to ensure the new point is as close to the original as possible
        while still being in K. '''

        # xtNew = max(xt - lambda)
        u = np.sort(y)[::-1] # Sort with largest first
        cumsum = np.cumsum(u) # cumulative sum array

        # top k entries are positive and get shifted by a constant, rest become 0.
        positivity = u * np.arange(1, len(u)+1)
        rho = positivity > (cumsum - 1.0)
        positiveIndices = np.nonzero(rho)[0]
        lastPositiveIdx = positiveIndices[-1]

        shift = (cumsum[lastPositiveIdx] - 1.0) / (lastPositiveIdx + 1.0)
        x = np.maximum(y - shift, 0.0)

        return x