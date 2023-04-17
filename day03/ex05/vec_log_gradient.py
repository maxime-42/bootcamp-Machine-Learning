
import numpy as np
from day03.ex01.log_pred import logistic_predict_
from utils.utils_function import check_x_and_theta, add_intercept


def vec_log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, without any for-loop. The three arrays must have compatible shapes.
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector (n +1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containg the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible shapes.
    Raises:
        This function should not raise any Exception.
    """
    if check_x_and_theta(x, theta) is False or x.size == 0:
        return None
    xp = add_intercept(x)
    y_hat = logistic_predict_(x, theta)
    
    if [isinstance(obj, np.ndarray)   for obj in [xp, y_hat]] is False :
        return None
    return 1/len(x) * xp.T.dot(y_hat - y) 