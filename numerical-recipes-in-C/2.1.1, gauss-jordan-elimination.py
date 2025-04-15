import numpy as np


def gauss_jordan_no_pivot(A, xs, bs):
    num_rows = len(A)
    A_inv = np.identity(num_rows)

    for n in range(num_rows):
        # print(f"n={n}: A={A}")
        # print(f"A^-1 = {A_inv}")

        pivot = A[n][n]
        A[n] /= pivot
        A_inv[n] /= pivot
        
        for m in range(num_rows):
            if m == n:
                continue

            scaling_factor = A[m][n]
            A[m] -= scaling_factor * A[n]
            A_inv[m] -= scaling_factor * A_inv[n]




    return A_inv

M = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
], dtype=float)


print(gauss_jordan_no_pivot(M, 0, 0))



    


