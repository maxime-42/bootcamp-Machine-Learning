import numpy as np

def add_polynomial_features(x, power):
    """
    Add polynomial features to matrix x by raising its columns to every power in the range of 1 up to the power given in argument.
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        power: has to be an int, the power up to which the columns of matrix x are going to be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of shape m * (np), containg the polynomial feature values for all training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """


    mod = x.shape[1]
    p = 2
    for i in range(power+2) :
       index = i % mod
       xn = x[:, index ] ** p
       x = np.hstack((x, xn.reshape(-1, 1)))
       if index == mod - 1:
           p += 1
       if p > power:
            return x
    return x

print("exemple 1:")
x = np.arange(1,11).reshape(5, 2)
print(add_polynomial_features(x, 4))
# print("exemple 2:")
# print(add_polynomial_features(x, 4))
