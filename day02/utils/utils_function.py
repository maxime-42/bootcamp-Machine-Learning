import numpy as np

def check_x_and_theta(x, theta):
    """
    Check requirement of argument y and x 
    Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    return 
        True if the requirement is fine
        False it mean there are some error in args
    """
    return isinstance(x, np.ndarray) and isinstance(theta, np.ndarray) and x.ndim == 2 and theta.ndim == 2 and x.shape[1] == theta.shape[0] - 1

def add_intercept(x=np.array([])):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return np.column_stack((np.ones(len(x)), x ))
