import numpy as np
import sys

def check_arg(x, power):
    """
    Checking type, Checking type of power
    Checking data type, 'i': signed integer, 'u': unsigned integer,'f': float
    """
    if not isinstance(x, np.ndarray):
        print("Unexpected type", file=sys.stderr)
        sys.exit(1)
    if x.dtype.kind not in ["i", "u", "f"]:
        print("Unexpected data type", file=sys.stderr)
        sys.exit(1)

    if (not isinstance(power, int)) or (power <= 0):
        print("Unexpected value for power", file=sys.stderr)
        sys.exit(1)

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
    if x.ndim == 1:
        x = np.expand_dims(x, axis=1)
    check_arg(x, power)
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
