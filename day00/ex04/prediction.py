"""import utils module"""
import numpy as np
from day00.ex02.prediction import simple_predict


def predict_(x=np.array([]), theta=np.array([])):
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

    y_hat = simple_predict(x, theta)
    return np.array([[i] for i in y_hat])


print("Exemple 2")
x = np.arange(1,6)
theta1 = np.array([[5], [0]])
print(predict_(x, theta1))
print("Exemple 3")
theta2 = np.array([[0], [1]])
print(predict_(x, theta2))
print("Exemple 3")
theta3 = np.array([[5], [3]])
print(predict_(x, theta3))
print("Exemple 4")
theta4 = np.array([[-3], [1]])
print(predict_(x, theta4))
