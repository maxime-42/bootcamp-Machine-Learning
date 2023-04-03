import numpy as np
from  day02.utils.utils_function import check_x_and_theta, add_intercept

def predict_(x:np.ndarray, theta:np.ndarray):
    """Computes the prediction vector y_hat from two non-empty numpy.array.
        Args:
        x: has to be an numpy.array, a vector of dimensions m * n.
        theta: has to be an numpy.array, a vector of dimensions (n + 1) * 1.
    Return:
        y_hat as a numpy.array, a vector of dimensions m * 1.
        None if x or theta are empty numpy.array.
        None if x or theta dimensions are not appropriate.
        None if x or theta is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    if check_x_and_theta(x, theta) is False:
        return None
    
    x = add_intercept(x)
    y_hat = x.dot(theta)
    # print(f"{y_hat}\n")
    return y_hat
