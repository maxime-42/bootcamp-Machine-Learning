"""import utils module"""
import numpy as np
import matplotlib.pyplot as plt
from day00.ex04.prediction import predict_

def simple_predict(x_1:np.ndarray, theta:np.ndarray):
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

def plot(x:np.ndarray, y:np.ndarray, theta:np.ndarray):
    """
    Plot the data and prediction line from three non-empty numpy.array.
    Args:
        x: has to be an numpy.array, a vector of dimension m * 1.
        y: has to be an numpy.array, a vector of dimension m * 1.
        theta: has to be an numpy.array, a vector of dimension 2 * 1.
    Returns:
        Nothing.
    Raises:
        This function should not raise any Exceptions.
    """

    y_hat = predict_(x, theta)
    
    plt.plot(x, y_hat)
    plt.scatter(x, y)

    plt.show()



x = np.arange(1,6)
y = np.array([3.74013816, 3.61473236, 4.57655287, 4.66793434, 5.95585554])
# Example 1:
theta1 = np.array([[4.5],[-0.2]])
plot(x, y, theta1)

# Example 2:
theta2 = np.array([[-1.5],[2]])
plot(x, y, theta2)
# Output


# Example 3:
theta3 = np.array([[3],[0.3]])
plot(x, y, theta3)
# Output