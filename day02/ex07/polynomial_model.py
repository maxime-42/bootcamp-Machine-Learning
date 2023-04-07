import numpy as np

def add_polynomial_features(x:np.ndarray, power:int):
    """
    Add polynomial features to vector x by raising its values up to the power given in argument.
        Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
            power: has to be an int, the power up to which the components of vector x are going to be raised.
        Return:
            The matrix of polynomial features as a numpy.array, of dimension m * n,
            containing the polynomial feature values for all training examples.
            None if x is an empty numpy.array.
            None if x or power is not of expected type.
        Raises:
            This function should not raise any Exception.
    """
    if isinstance(x, np.ndarray) is False or x.shape == (x.size, 1) is False:
        return None
    matrix = [x**j for j in range(1, power+1)]
    return np.column_stack(matrix)

# x = np.arange(1,6).reshape(-1, 1)
# print("exempl 0:")
# print(add_polynomial_features(x, 3))
# print("exempl 1:")
# print(add_polynomial_features(x, 6))
