import numpy as np
from utils.utils_function import  check_x_and_theta, add_intercept
from day04.ex01.l2_reg import  l2
from day03.ex00.sigmoid import sigmoid_


def reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, with two for-loops. The three array
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    new_theta = np.zeros(theta.shape)
    for  x_, y_true in zip(x, y):
        y_hat =  sigmoid_(np.dot(x_, theta[1:]) + theta[0])
        new_theta[0] += y_hat - y_true
        new_theta[1:] += (x_ * (y_hat - y_true)).reshape(-1, 1)

    new_theta[1:] +=  lambda_ * theta[1:]

    return new_theta * 1/len(y)

def vec_reg_logistic_grad(y, x, theta, lambda_):
    """Computes the regularized logistic gradient of three non-empty numpy.ndarray, without any for-loop. The three arr
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of shape m * n.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        A numpy.ndarray, a vector of shape n * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
    Raises:
        This function should not raise any Exception.
    """
    if not check_x_and_theta(x, theta):
        return None
    if all([isinstance(obj, np.ndarray) and len(obj) > 1 for obj in  (y, theta)]) is False:
        return None
    x = add_intercept(x)
    y_hat =  sigmoid_(np.dot(x, theta))
    theta_cp = np.copy(theta)
    theta_cp[0] = 0
    reg = theta_cp * lambda_ 
    res = x.T @ (y_hat - y) + reg
    return res * 1/len(y)
