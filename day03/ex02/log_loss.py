import numpy as np

def log_loss_(y, y_hat, eps=1e-15):
    """
    Computes the logistic loss value.
    Args:
        y: has to be an numpy.ndarray, a vector of shape m * 1.
        y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
        eps: has to be a float, epsilon (default=1e-15)
    Returns:
        The logistic loss value as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    if all([isinstance(item, np.ndarray) and item == (item.size, 1)  ]  for item in [y, y_hat]) is False :
        return None
    
    return -1/len(y) * np.sum(y * np.log(y_hat)  + (1-y) * np.log(1-y_hat))