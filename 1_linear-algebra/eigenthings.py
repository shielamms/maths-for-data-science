from numpy import array, diag, zeros, concatenate
from numpy.linalg import eig, inv


def confirm_eigenvector(matrix, eigenvector, eigenvalue):
    assert eigenvector.all()
    assert eigenvalue

    scaled_matrix = matrix.dot(eigenvector)
    scaled_vector = eigenvector * eigenvalue
    result = (scaled_matrix.round(8) == scaled_vector.round(8))
    return result.all()

def reconstruct_matrix_from_eigenthings(eigenvectors, eigenvalues):
    vector_inverse = inv(eigenvectors)
    values_identity = diag(eigenvalues)
    return eigenvectors.dot(values_identity).dot(vector_inverse)

def test_confirm_eigenvectors():
    print('--- test_confirm_eigenvectors ---')

    matrix = array([[1,2],[3,4]])
    values, vectors = eig(matrix)
    expected = [True, True]

    print('Matrix:\n', matrix)
    print('Test vectors:\n', vectors)
    print('Test values:\n', values)
    print()

    for i in range(values.size):
        actual = confirm_eigenvector(matrix, vectors[:, i], values[i])
        print(f'Is vector {i+1} an eigenvector? ', actual)
        print(f'Expected: ', expected[i])
        print()

def test_reconstruct_matrix_from_eigenthings():
    print('\n--- test_reconstruct_matrix_from_eigenthings ---')

    original_matrix = array([[1,2],[3,4]])
    eigenvalues, eigenvectors = eig(original_matrix)

    reconstructed_matrix = reconstruct_matrix_from_eigenthings(
                                eigenvectors,
                                eigenvalues)
    print('Original matrix:\n', original_matrix)
    print('Reconstructed matrix:\n', reconstructed_matrix)
    print(original_matrix.round(8) == reconstructed_matrix.round(8))

if __name__ == '__main__':
    test_confirm_eigenvectors()
    test_reconstruct_matrix_from_eigenthings()
