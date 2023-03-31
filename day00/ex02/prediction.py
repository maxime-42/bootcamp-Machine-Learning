"""import utils module"""
import numpy as np

def simple_predict(x_1=np.array([]), theta=np.array([])):
    """
    Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
    Args:
        x: has to be an numpy.ndarray, a vector of dimension m * 1.
        theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
    Returns:
        y_hat as a numpy.ndarray, a vector of dimension m * 1.
        None if x or theta are empty numpy.ndarray.
        None if x or theta dimensions are not appropriate.
    Raises:
    This function should not raise any Exception.
    """

    if x_1.size == 0 or theta.size == 0:
        return None
    x_0 = np.ones(len(x_1))
    x = np.column_stack((x_0, x_1 ))
    return x.dot(theta)

x = np.arange(1,6)
theta2 = np.array([-3, 1])
r = simple_predict(x, theta2)
print(r)
