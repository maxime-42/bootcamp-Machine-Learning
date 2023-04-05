import numpy as np

from day02.ex01.prediction import predict_
from  day02.utils.utils_function import check_x_and_theta, add_intercept

def gradient(x:np.ndarray, y:np.ndarray, theta:np.ndarray):
    """
        Computes a gradient vector from three non-empty numpy.array, without any for-loop.
        The three arrays must have the compatible dimensions.
            Args:
            x: has to be an numpy.array, a matrix of dimension m * n.
            y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector (n +1) * 1.
        Return:
            The gradient as a numpy.array, a vector of dimensions n * 1,
            containg the result of the formula for all j.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible dimensions.
            None if x, y or theta is not of expected type.
        Raises:
            This function should not raise any Exception.
    """

    if check_x_and_theta(x, theta) is False:
        return None

    y_hat = predict_(x, theta)
    m = len(x)
    x = add_intercept(x)
    return x.T.dot(y_hat - y) * 1/m