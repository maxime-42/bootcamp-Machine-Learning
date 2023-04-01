"""import utils module"""
import numpy as np

from day00.utils.utils_function import check_x_and_theta 


def simple_predict(x=np.array([]), theta=np.array([])):
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """
    if not check_x_and_theta(x, theta):
        return None

    x = np.column_stack((np.ones(len(x)), x ))
   
    tmp = theta[0] + theta[1] * x
    y_hat = np.delete(tmp, 0, 1).transpose()
    return y_hat[0]
