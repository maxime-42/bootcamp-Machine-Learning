"""Understand and manipulate the notion of gradient and gradient descent in machine learning. 
You must write a function that computes the gradient of the loss function. It must
compute a partial derivative with respect to each theta parameter separately, and return
the vector gradient.
"""
import numpy as np


def check_requirement_args(x, y, theta):
    """verify the expected of arguments x, y theta"""
    if  all(isinstance(obj, np.ndarray) and obj.shape in [(obj.size, 1)] for  obj in [x, y, theta]) is False : 
        return False
    if theta.shape != (2, 1) or y.size != x.size:
        return False
    return True

def simple_predict(x=np.array([]), theta=np.array([])):
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

    x_0 = np.ones(len(x))
    return np.column_stack((x_0, x )).dot(theta)


def simple_gradient(x, y, theta):
    """
        Computes a gradient vector from three non-empty numpy.array, with a for-loop.
        The three arrays must have compatible shapes.
        Args:
            x: has to be an numpy.array, a vector of shape m * 1.
            y: has to be an numpy.array, a vector of shape m * 1.
            theta: has to be an numpy.array, a 2 * 1 vector.
        Return:
            The gradient as a numpy.array, a vector of shape 2 * 1.
            None if x, y, or theta are empty numpy.array.
            None if x, y and theta do not have compatible shapes.
            None if x, y or theta is not of the expected type.
        Raises:
            This function should not raise any Exception.
    """

    if check_requirement_args(x, y, theta) is False:
        return None
    y_hat = simple_predict(x, theta)
    if y_hat is None:
        return None
    m = len(x)
    result = np.zeros(2)
    for i in range(m):
        result[0] += y_hat[i] - y[i]
        result[1] += (y_hat[i] - y[i]) * x[i]
    return 1/m * result 

x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1) )
y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))


theta1 = np.array([2, 0.7]).reshape((-1, 1))
print(simple_gradient(x, y, theta1))
