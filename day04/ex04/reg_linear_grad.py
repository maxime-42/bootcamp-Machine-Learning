import numpy as np
from utils.utils_function import  check_x_and_theta, add_intercept
from day04.ex01.l2_reg import  l2


def reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    with two for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """

    if not check_x_and_theta(x, theta):
        return None
    if all([isinstance(obj, np.ndarray) and len(obj) > 1 for obj in  (y, theta)]) is False:
        return None

    new_theta = np.zeros(theta.shape)
    for y_true, x_features in zip(y, x):
        y_hat = ( x_features @ theta[1:]) + theta[0]
        new_theta[0] += y_hat - y_true
        new_theta[1:] += (x_features * (y_hat - y_true)).reshape(-1, 1)
    new_theta[1:] += lambda_ * theta[1:]
    return (new_theta) / 1/len(y)




def vec_reg_linear_grad(y, x, theta, lambda_):
    """Computes the regularized linear gradient of three non-empty numpy.ndarray,
    without any for-loop. The three arrays must have compatible shapes.
    Args:
        y: has to be a numpy.ndarray, a vector of shape m * 1.
        x: has to be a numpy.ndarray, a matrix of dimesion m * n.
        theta: has to be a numpy.ndarray, a vector of shape (n + 1) * 1.
        lambda_: has to be a float.
    Return:
        A numpy.ndarray, a vector of shape (n + 1) * 1, containing the results of the formula for all j.
        None if y, x, or theta are empty numpy.ndarray.
        None if y, x or theta does not share compatibles shapes.
        None if y, x or theta or lambda_ is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if not check_x_and_theta(x, theta):
        return None
    if all([isinstance(obj, np.ndarray) and len(obj) > 1 for obj in  (y, theta)]) is False:
        return None
    x = add_intercept(x)
    
    pred = x.dot(theta)
    # is used to set the bias term of the theta vector to zero.
    # This is done because the regularization term only applies 
    # to the non-bias terms of theta,
    # and setting the bias term to zero ensures that it is not affected by the regularization term.
    theta[0] = 0

    reg = lambda_ * theta
    new_theta = x.T.dot((pred - y)) + reg
    return new_theta * 1/len(y)