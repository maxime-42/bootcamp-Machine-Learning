import numpy as np
from day04.ex01.l2_reg import  l2

def reg_log_loss_(y, y_hat, theta, lambda_):
    eps=1e-15
    """Computes the regularized loss of a logistic regression model from two non-empty numpy.ndarray, without any for l
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta is empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    if all([isinstance(obj, np.ndarray) and len(obj) > 1 for obj in  (y, y_hat, theta)]) is False or y.shape != y_hat.shape:
        return None
    if not isinstance(lambda_, float):
        return None
    new_theta = np.squeeze(theta[1:])
    y = np.squeeze(y)
    y_hat_ = np.squeeze(y_hat)
    loss = y @ np.log(y_hat_ + eps) + (1 - y) @ np.log(1 - y_hat_ + eps)
    reg = lambda_ * new_theta @ new_theta / (2 * y.shape[0])
    return -loss / y.shape[0] + reg