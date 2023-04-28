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
    """Add polynomial features to vector x by raising its values up to the power given in
        argument.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        power: has to be an int, the power up to which the components of vector x are going to
            be raised.
    Returns:
        The matrix of polynomial features as a numpy.ndarray, of dimension m * n, containg he
            polynomial feature values for all training examples.
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    return np.concatenate([x ** (i + 1) for i in range(power)], axis=1)
