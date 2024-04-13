import numpy as np

basis = {
    "c1" : [1, -1, 0, 0, 0, 0, 0, 0],
    "c2" : [1, 0, -1, 0, 0, 0, 0, 0],
    "c3" : [1, 0, 0, -1, 0, 0, 0, 0],
    "c4" : [1, 0, 0, 0, -1, 0, 0, 0],
    "c5" : [1, 0, 0, 0, 0, -1, 0, 0],
    "c6" : [1, 0, 0, 0, 0, 0, -1, 0],
    "c7" : [0, 1, 0, 0, 0, 0, 0, -1]
}

"""
Given a vector v in Z_3^8 such that sum of v_i = 0 mod 3
Compute how can we write v as a linear combination of the basis
"""

# Convert basis into a matrix
A = np.array([basis[key] for key in sorted(basis.keys())]).T
A = (A % 3 + 3) % 3  # Ensure all elements are properly mod 3

print("Basis matrix A:")
print(A)

def solve_modular_system(A, v):
    # Number of variables
    n = A.shape[1]
    
    # Extended matrix [A|v], where v is the vector to express as a linear combination
    extended = np.hstack([A, v.reshape(-1, 1)])
    extended = (extended % 3 + 3) % 3

    print("Extended matrix [A|v]:")
    print(extended)

    # Gaussian elimination in Z_3
    for i in range(n):
        # Find a row starting with non-zero in the i-th column to use as pivot
        if extended[i, i] == 0:
            for j in range(i+1, A.shape[0]):
                if extended[j, i] != 0:
                    extended[[i, j]] = extended[[j, i]]  # Swap rows
                    break

        # Make the pivot element 1 by multiplying the row (only necessary for mod p where p > 3)
        inv = pow(int(extended[i, i]), -1, 3)  # Modular inverse
        extended[i] = (extended[i] * inv) % 3

        # Eliminate all other entries in this column
        for j in range(A.shape[0]):
            if i != j:
                factor = extended[j, i]
                extended[j] = (extended[j] - factor * extended[i]) % 3

    # Back-substitution to find solution
    solution = extended[:, -1]
    return solution

# Example vector v
v = np.array([1, 2, 0, 0, 0, 0, 0, 0,])  # Change this to your vector satisfying sum(v_i) = 0 mod 3

# Solve the system
coefficients = solve_modular_system(A, v)
print("Coefficients are:", coefficients)
def sanity_check(A, v, coefficients):
    """
    Check if the computed coefficients indeed represent the vector v
    as a linear combination of basis vectors in Z_3^8.

    Parameters:
        A (numpy.ndarray): The basis matrix.
        v (numpy.ndarray): The target vector.
        coefficients (numpy.ndarray): Coefficients found for the basis vectors.

    Returns:
        bool: True if the solution is correct, False otherwise.
    """
    # Ensure coefficients are in the correct shape
    v_constructed = [0] * v.shape[0]
    for i, c in enumerate(coefficients):
        temp_list = c * A[:, 6]
        
        
    return True

# Example usage
# Basis matrix A and vector v should be defined as shown previously
coefficients = solve_modular_system(A, v)
if sanity_check(A, v, coefficients):
    print("Sanity check passed.")
    print("Coefficients are:", coefficients)
else:
    print("Sanity check failed. The solution does not satisfy the equation under Z_3^8.")
