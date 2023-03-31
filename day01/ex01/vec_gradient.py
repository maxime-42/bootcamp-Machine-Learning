import numpy as np

from day01.ex00.gradient import check_requirement_args, simple_predict

def gradient(x, y, theta):
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

    xt = np.column_stack((np.ones(len(x)), x )).transpose()
    m = len(x)
    
    return 1/m * xt.dot(y_hat - y)


# x = np.array([12.4956442, 21.5007972, 31.5527382, 48.9145838, 57.5088733]).reshape((-1, 1))
# y = np.array([37.4013816, 36.1473236, 45.7655287, 46.6793434, 59.5585554]).reshape((-1, 1))
# print("Example 0:")
# theta1 = np.array([2, 0.7]).reshape((-1, 1))
# print(gradient(x, y, theta1))

# print("Example 1:")
# theta2 = np.array([1, -0.4]).reshape((-1, 1))
# print(gradient(x, y, theta2))
# # Output:
# # array([[-57.8682...], [-2230.1229...]])