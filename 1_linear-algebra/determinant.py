import copy
import numpy as np
from numpy import array

def get_determinant(matrix):
    if matrix.shape[0] == 1:
        return matrix.shape[0]
    elif matrix.shape[0] == 2:
        return (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])
    else:
        determinant = 0

        for col in range(matrix.shape[1]):
            minor = copy.deepcopy(matrix)

            # remove the first row (axis=0)
            minor = np.delete(minor, 0, 0)
            # remove the col-th column (axis=1)
            minor = np.delete(minor, col, 1)

            cofactor_sign = ((-1)**col)
            cofactor = get_determinant(minor)
            determinant += cofactor_sign * matrix[0][col] * cofactor

        return determinant


if __name__ == '__main__':
    A = array([[1,2],[3,4]])
    print('Matrix:\n', A)
    print('Determinant: ', get_determinant(A))
