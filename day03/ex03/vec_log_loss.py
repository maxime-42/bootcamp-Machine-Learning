import numpy as np

def vec_log_loss_(y, y_hat, eps=1e-15):
    """
        Compute the logistic loss value.
        Args:
            y: has to be an numpy.ndarray, a vector of shape m * 1.
            y_hat: has to be an numpy.ndarray, a vector of shape m * 1.
            eps: epsilon (default=1e-15)
        Returns:
            The logistic loss value as a float.
            None on any error.
        Raises:
            This function should not raise any Exception.
    """

    test = all([isinstance(item, np.ndarray) and item == (item.size, 1)  ]  for item in [y, y_hat])
    if False in (test, y.shape )  :
        return None
    log_loss = np.dot(y.T, np.log(y_hat + eps))   + np.dot((1 - y).T, np.log(1 - y_hat + eps))
    return - float(log_loss) / y.shape[0]
