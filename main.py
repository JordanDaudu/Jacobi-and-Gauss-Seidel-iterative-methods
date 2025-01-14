# Jacobi and Gauss-Seidel Iterative Methods

class ConvergenceError(Exception):
    """Exception for handling non-convergence in iterative methods."""
    pass

def norm(vector):
    """Computes the infinity norm of a vector."""
    return max(abs(v) for v in vector)

def print_iteration_header(A, verbose=True):
    """
    Prints the header for the iteration table and checks if the matrix is diagonally dominant.

    Parameters:
        A (list of lists): Coefficient matrix.
        verbose (bool): If True, prints additional diagnostic information.
    """
    n = len(A)

    if is_diagonally_dominant(A) and verbose:
        print("Matrix is diagonally dominant.")
    if not is_diagonally_dominant(A):
        print("Matrix is not diagonally dominant. Attempting to modify the matrix...")
        A = make_diagonally_dominant(A)
        if is_diagonally_dominant(A) and verbose:
            print("Matrix modified to be diagonally dominant:\n", A)

    if verbose:
        print("Iteration" + "\t\t\t".join([" {:>12}".format(f"x{i + 1}") for i in range(n)]))
        print("--------------------------------------------------------------------------------")

def is_diagonally_dominant(A):
    """
    Checks if a matrix is diagonally dominant.

    Parameters:
        A (list of lists): The matrix to check.

    Returns:
        bool: True if the matrix is diagonally dominant, False otherwise.
    """
    for i in range(len(A)):
        row_sum = sum(abs(A[i][j]) for j in range(len(A)) if j != i)
        if abs(A[i][i]) < row_sum:
            return False
    return True

def make_diagonally_dominant(A):
    """
    Modifies the matrix A to make it diagonally dominant by swapping rows if necessary.

    Parameters:
        A (list of lists): The coefficient matrix to be modified.

    Returns:
        list of lists: The modified matrix that is diagonally dominant (if possible).
    """
    n = len(A)

    for i in range(n):
        if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(n) if j != i):
            # Find a row with a larger diagonal element
            for j in range(i + 1, n):
                if abs(A[j][i]) > abs(A[i][i]):
                    # Swap row i and row j
                    A[i], A[j] = A[j], A[i]
                    break
            # After attempting to swap, if no dominant diagonal is found, print a warning
            if abs(A[i][i]) < sum(abs(A[i][j]) for j in range(n) if j != i):
                print(f"Warning: Row {i} still not diagonally dominant after attempting row swaps.")

    return A

def jacobi_iterative(A, b, X0=None, TOL=0.00001, N=200, verbose=True):
    """
    Performs Jacobi iterative method to solve the system of linear equations Ax = b.

    This method is an iterative approach to find an approximate solution to a
    system of linear equations. It is particularly useful for large, sparse matrices
    and when the coefficient matrix `A` is diagonally dominant.

    Parameters:
        A (list of lists): Coefficient matrix of size n x n.
        b (list): Solution vector size n.
        X0 (list, optional): Initial guess for the solution vector. Defaults to a zero vector.
        TOL (float, optional): Tolerance for the convergence criterion. Defaults to 0.00001.
        N (int, optional): Maximum number of iterations allowed. Defaults to 200.
        verbose (bool, optional): If True, prints iteration details. Defaults to True.

    Returns:
        list: Approximate solution vector `x` of size n.

    Raises:
        ConvergenceError: If the method fails to converge within the maximum number of iterations.

    Notes:
        - The convergence of the Jacobi method is guaranteed if `A` is strictly
          diagonally dominant. If the matrix is not diagonally dominant, convergence
          is not guaranteed but may still occur in some cases.
        - The norm of the difference between successive approximations is used
          as the convergence criterion.
    """
    n = len(A)
    if X0 is None:
        X0 = [0.0] * n

    print_iteration_header(A, verbose)

    for k in range(1, N + 1):
        x = [0.0] * n
        for i in range(n):
            sigma = sum(A[i][j] * X0[j] for j in range(n) if j != i)
            x[i] = (b[i] - sigma) / A[i][i]

        if verbose:
            print(f"{k:<15}" + "\t\t".join(f"{val:<15.10f}" for val in x))

        if norm([x[i] - X0[i] for i in range(n)]) < TOL:
            if not is_diagonally_dominant(A):
                print("\n|Warning: Matrix is not diagonally dominant, but the solution is within tolerance and converged.|")
            return x

        X0 = x.copy()

    print("Maximum number of iterations exceeded, Matrix is not converging")
    raise ConvergenceError("Jacobi method failed to converge within the maximum number of iterations.")

def gauss_seidel(A, b, X0=None, TOL=0.00001, N=200, verbose=True):
    """
    Performs Gauss-Seidel iterations to solve the system of linear equations Ax = b.

    Parameters:
        A (list of lists): Coefficient matrix of size n x n.
        b (list): Solution vector size n.
        X0 (list, optional): Initial guess for the solution. Defaults to a zero vector.
        TOL (float, optional): Tolerance for convergence. Defaults to 0.00001.
        N (int, optional): Maximum number of iterations. Defaults to 200.
        verbose (bool, optional): If True, prints iteration details. Defaults to True.

    Returns:
        list: Approximate solution vector.

    Raises:
        ConvergenceError: If the method fails to converge within the maximum number of iterations.

    Notes:
        - The Gauss-Seidel method updates each component of the solution vector
          immediately after it is computed, unlike the Jacobi method, which updates
          all components simultaneously at the end of each iteration.
        - The convergence of the method is guaranteed if the coefficient matrix `A` is
          strictly diagonally dominant or symmetric positive definite.
        - The norm of the difference between successive approximations is used
          as the convergence criterion.
        - If the matrix `A` is not diagonally dominant, the method may still converge
          in some cases, but this is not guaranteed. A warning will be displayed if
          convergence occurs despite the lack of diagonal dominance.
    """
    n = len(A)
    if X0 is None:
        X0 = [0.0] * n

    print_iteration_header(A, verbose)

    for k in range(1, N + 1):
        x = X0.copy()
        for i in range(n):
            sigma1 = sum(A[i][j] * x[j] for j in range(i))
            sigma2 = sum(A[i][j] * X0[j] for j in range(i + 1, n))
            x[i] = (b[i] - sigma1 - sigma2) / A[i][i]

        if verbose:
            print(f"{k:<15}" + "\t\t".join(f"{val:<15.10f}" for val in x))

        if norm([x[i] - X0[i] for i in range(n)]) < TOL:
            if not is_diagonally_dominant(A):
                print("\n|Warning: Matrix is not diagonally dominant, but the solution is within tolerance and converged.|")
            return x

        X0 = x.copy()

    print("Maximum number of iterations exceeded, Matrix is not converging")
    raise ConvergenceError("Gauss-Seidel method failed to converge within the maximum number of iterations.")

if __name__ == "__main__":
    matrixA = [[4, 2, 0], [2, 10, 4], [0, 4, 5]]
    vectorB = [2, 6, 5]
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
    try:
        if choice == 1:
            solution = jacobi_iterative(matrixA, vectorB, verbose=True)
        else:
            solution = gauss_seidel(matrixA, vectorB, verbose=True)
        print("\nApproximate solution:", solution)
    except ConvergenceError as message:
        print("\nError:", message)
        exit(0)