"""import utils module"""
import numpy as np

# from day00.ex02.prediction import check_x_and_theta

def add_intercept(x=np.array([])):
    """Adds a column of 1’s to the non-empty numpy.array x.
    Args:
        x: has to be a numpy.array of dimension m * n.
    Returns:
        X, a numpy.array of dimension m * (n + 1).
        None if x is not a numpy.array.
        None if x is an empty numpy.array.
    Raises:
        This function should not raise any Exception.
    """
    if not isinstance(x, np.ndarray) or x.size == 0:
        return None
    if len(x.shape) == 1:
        x = x.reshape(-1, 1)
    return np.column_stack((np.ones(len(x)), x ))


# def simple_predict(x=np.array([]), theta=np.array([])):
#     """
#     Computes the vector of prediction y_hat from two non-empty numpy.ndarray.
#     Args:
#         x: has to be an numpy.ndarray, a vector of dimension m * 1.
#         theta: has to be an numpy.ndarray, a vector of dimension 2 * 1.
#     Returns:
#         y_hat as a numpy.ndarray, a vector of dimension m * 1.
#         None if x or theta are empty numpy.ndarray.
#         None if x or theta dimensions are not appropriate.
#     Raises:
#     This function should not raise any Exception.
#     """

#     if check_x_and_theta(x, theta) == False:
#         return None

#     x = np.column_stack((np.ones(len(x)), x ))
#     return x.dot(theta)

print("Exemple 1")
x = np.arange(1,6)
print(add_intercept(x))

y = np.arange(1,10).reshape((3,3))
print("Exemple 2")
print(add_intercept(y))
# theta2 = np.array([-3, 1])
# r = simple_predict(x, theta2)
# print(r)
# check_x_and_theta(x, theta2)