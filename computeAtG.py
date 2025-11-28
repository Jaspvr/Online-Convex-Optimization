import numpy as np

def commputeAtG(A, grad, p):
    """ONS updates with A^{-1}, AdaGrad updates with A^{-1/2}
    here we parametrize the exponent, giving A^{p}@grad"""

    # Use A^p = Q Λ^p Q^T
    eigenValues, eigenVectors = np.linalg.eigh(A)
    eigenValues = np.clip(eigenValues, 1e-10, None)
    eigenValuesP = eigenValues ** p

    qtgrad = eigenVectors.T @ grad
    qtGradVals = eigenValuesP * qtgrad

    return eigenVectors @ qtGradVals
