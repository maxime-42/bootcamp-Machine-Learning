import numpy as np
import sys

def iterative_l2(theta):
    """Computes the L2 regularization of a non-empty numpy.ndarray, with a for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray) or  theta.shape != (theta.size, 1):
        return None
    res = 0.0
    for i in range(1 , len(theta)):
        res += theta[i]  **2
    return res[0]


def l2(theta:np.array):
    """Computes the L2 regularization of a non-empty numpy.ndarray, without any for-loop.
    Args:
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
    Returns:
        The L2 regularization as a float.
        None if theta in an empty numpy.ndarray.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(theta, np.ndarray) or  theta.shape != (theta.size, 1):
        print("Unexpected data or shape of theta", file=sys.stdout)
        return None
    res = np.dot(theta[1:].T, theta[1:])
    return  float(res[0][0])