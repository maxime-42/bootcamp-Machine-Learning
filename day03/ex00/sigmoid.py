import numpy as np


def sigmoid_(x:np.ndarray):
    """
    Compute the sigmoid of a vector.
    Args:
        x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
        The sigmoid value as a numpy.ndarray of shape (m, 1).
        None if x is an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if isinstance(x, np.ndarray) is False or len(x) == 0:
        return None
    return np.divide(1., 1. + np.exp(-x))
