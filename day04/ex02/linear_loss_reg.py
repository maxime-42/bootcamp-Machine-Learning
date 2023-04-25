import numpy as np
from day04.ex01.l2_reg import  l2

def reg_loss_(y, y_hat, theta, lambda_):
    """Computes the regularized loss of a linear regression model from two non-empty numpy.array, without any for loop. The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        theta: has to be a numpy.ndarray, a vector of shape n * 1.
        lambda_: has to be a float.
    Returns:
        The regularized loss as a float.
        None if y, y_hat, or theta are empty numpy.ndarray.
        None if y and y_hat do not share the same shapes.
    Raises:
        This function should not raise any Exception.
    """
    if all([isinstance(obj, np.ndarray) and len(obj) > 1 for obj in  (y, y_hat, theta)]) is False or y.shape != y_hat.shape:
        return None
    if not isinstance(lambda_, float):
        return None
    loss = (y - y_hat).T @ (y - y_hat)
    
    reg = lambda_ * l2(theta)
    return float(0.5 * (loss + reg) / y.shape[0])