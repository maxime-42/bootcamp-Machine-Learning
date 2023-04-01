import numpy as np

def check_x_and_theta(x, theta):
    """
    Check requirement of argument y and x 
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    return 
        True if the requirement is fine
        False it mean there are some error in args
    """

    return isinstance(x, np.ndarray) and (x.shape == (x.shape[0], ) or (x.shape[0], 1 )) and \
           isinstance(theta, np.ndarray) and (theta.shape == (2, 1) or theta.shape == (2, ))
