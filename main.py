
# Jacobi Iterative Method

import numpy as np
from numpy.linalg import norm

def is_diagonally_dominant(A):
    """Check if a matrix is diagonally dominant."""
    for i in range(len(A)):
        row_sum = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
        if abs(A[i][i]) < row_sum:
            return False
    return True

def jacobi_iterative(A, b, X0=None, TOL=1e-16, N=200, verbose=True):
    """
    Performs Jacobi iterations to solve Ax = b.

    Parameters:
        A (ndarray): Coefficient matrix.
        b (ndarray): Solution vector.
        X0 (ndarray): Initial guess for the solution. Defaults to a zero vector.
        TOL (float): Tolerance for convergence. Defaults to 1e-16.
        N (int): Maximum number of iterations. Defaults to 200.
        verbose (bool): If True, prints iteration details.

    Returns:
        ndarray: Approximate solution vector.
    """
    n = len(A)
    if X0 is None:
        X0 = np.zeros_like(b, dtype=np.double)

    if not is_diagonally_dominant(A):
        print("Warning: Matrix is not diagonally dominant. Convergence is not guaranteed.")

    if verbose:
        print("Iteration" + "\t\t\t".join([" {:>12}".format(f"x{i+1}") for i in range(n)]))
        print("--------------------------------------------------------------------------------")

    for k in range(1, N + 1):
        x = np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma = sum(A[i][j] * X0[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

        if verbose:
            print(f"{k:<15}" + "\t\t".join(f"{val:<15.10f}" for val in x))

        if norm(x - X0, np.inf) < TOL:
            return x

        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return x


if __name__ == "__main__":
    A = np.array([[1, 0, -2], [0, 1, 0], [0, 0, 4/3]])
    b = np.array([-1, 1, 4])

    solution = jacobi_iterative(A, b, verbose=True)
    print("\nApproximate solution:", solution)



