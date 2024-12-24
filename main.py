# Jacobi and Gauss-Seidel Iterative Methods

import numpy as np
from numpy.linalg import norm

def is_diagonally_dominant(A):
    """Check if a matrix is diagonally dominant."""
    for i in range(len(A)):
        row_sum = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
        if abs(A[i][i]) < row_sum:
            return False
    return True


def make_diagonally_dominant(A):
    """
    Modifies the matrix A to make it diagonally dominant by swapping rows if necessary.

    Parameters:
        A (ndarray): The coefficient matrix to be modified.

    Returns:
        ndarray: The modified matrix that is diagonally dominant (if possible).
    """
    n = len(A)

    for i in range(n):
        if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
            # Find a row with a larger diagonal element
            for j in range(i + 1, n):
                if abs(A[j, i]) > abs(A[i, i]):
                    # Swap row i and row j
                    A[[i, j]] = A[[j, i]]
                    break
            # After attempting to swap, if no dominant diagonal is found, print a warning
            if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
                print(f"Warning: Row {i} still not diagonally dominant after attempting row swaps.")

    return A


def jacobi_iterative(A, b, X0=None, TOL=0.00001, N=200, verbose=True):
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

    if is_diagonally_dominant(A) and verbose:
        print("Matrix is diagonally dominant.")
    if not is_diagonally_dominant(A):
        print("Matrix is not diagonally dominant. Attempting to modify the matrix...")
        A = make_diagonally_dominant(A)
        if is_diagonally_dominant(A) and verbose:
            print("Matrix modified to be diagonally dominant:\n", A)

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
            if not is_diagonally_dominant(A):
                print("Warning: Matrix is not diagonally dominant, but the solution is within tolerance and converged.")
            return x

        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return x


def gauss_seidel(A, b, X0=None, TOL=0.00001, N=200, verbose=True):
    """
    Performs Gauss-Seidel iterations to solve Ax = b.

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

    if is_diagonally_dominant(A) and verbose:
        print("Matrix is diagonally dominant.")
    if not is_diagonally_dominant(A):
        print("Matrix is not diagonally dominant. Attempting to modify the matrix...")
        A = make_diagonally_dominant(A)
        if is_diagonally_dominant(A) and verbose:
            print("Matrix modified to be diagonally dominant:\n", A)

    if verbose:
        print("Iteration" + "\t\t\t".join([" {:>12}".format(f"x{i+1}") for i in range(n)]))
        print("--------------------------------------------------------------------------------")

    for k in range(1, N + 1):
        x = np.zeros(n, dtype=np.double)
        for i in range(n):
            sigma1 = sum(A[i][j] * x[j] for j in range(i))
            sigma2 = sum(A[i][j] * X0[j] for j in range(i + 1, n))
            x[i] = (b[i] - sigma1 - sigma2) / A[i][i]

        if verbose:
            print(f"{k:<15}" + "\t\t".join(f"{val:<15.10f}" for val in x))

        if norm(x - X0, np.inf) < TOL:
            if not is_diagonally_dominant(A):
                print("Warning: Matrix is not diagonally dominant, but the solution is within tolerance and converged.")
            return x

        X0 = x.copy()

    print("Maximum number of iterations exceeded")
    return x


if __name__ == "__main__":
    A = np.array([[4, 2, 0], [2, 10, 4], [0, 4, 5]])
    b = np.array([2, 6, 5])
    while True:
        print("Please choose the method you want to use:")
        print("1. Jacobi Iterative Method")
        print("2. Gauss-Seidel Iterative Method")
        try:
            choice = int(input("Enter your choice: "))
            if choice in [1, 2]:
                break
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            print("Invalid input. Please enter a number (1 or 2).")

    print("================================================================================")
    if choice == 1:
        solution = jacobi_iterative(A, b, verbose=True)
    else:
        solution = gauss_seidel(A, b, verbose=True)
    print("\nApproximate solution:", solution)