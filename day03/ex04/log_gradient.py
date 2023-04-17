import numpy as np
from day03.ex01.log_pred import logistic_predict_
from utils.utils_function import check_x_and_theta


def log_gradient(x, y, theta):
    """
    Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatible dimensions.
    Args:
        x: has to be an numpy.ndarray, a matrix of shape m * n.
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
        The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
        None if x, y, or theta are empty numpy.ndarray.
        None if x, y and theta do not have compatible dimensions.
    Raises:
        This function should not raise any Exception.
    """
    if check_x_and_theta(x, theta) is False or x.size == 0:
        return None
    res = np.zeros(theta.shape)
    m, n = x.shape
    for i in range(m):
        y_hat = logistic_predict_(x, theta)
        res[0][0] += 1/len(y) * np.sum(y_hat[i] - y[i])
        for j in range(n): 
           res[j + 1][0] += 1/len(y) * np.sum(y_hat[i] - y[i]) * x[i][j]

    return res