""""""

import numpy as np

def predict_(x_1:np.ndarray=np.array([]), theta:np.ndarray=np.array([])):
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

def loss_elem_(y:np.ndarray, y_hat:np.ndarray):
    """
    Description:
        Calculates all the elements (y_pred - y)^2 of the loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_elem: numpy.array, a vector of dimension (number of the training examples,1).
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """

    if isinstance(y, np.ndarray) is False or isinstance(y_hat, np.ndarray) is False:
         return None
    if y_hat.shape != y.shape:
        return None
    return (y_hat - y)**2


def loss_(y:np.ndarray, y_hat:np.ndarray):
    """
    Description:
        Calculates the value of loss function.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        J_value : has to be a float.
        None if there is a dimension matching problem between X, Y or theta.
        None if any argument is not of the expected type.
    Raises:
        This function should not raise any Exception.
    """
    if isinstance(y, np.ndarray) is False or isinstance(y_hat, np.ndarray) is False:
         return None
    squared_distances  = loss_elem_(y, y_hat)
    divizer = 2*len(y)
    return (1/divizer) * np.sum(squared_distances)


# x1 = np.array([[0.], [1.], [2.], [3.], [4.]])
# theta1 = np.array([[2.], [4.]])
# y_hat1 = predict_(x1, theta1)
# y1 = np.array([[2.], [7.], [12.], [17.], [22.]])

# # Example 1:
# print(loss_elem_(y1, y_hat1))

# #exemple 2
# print(loss_(y1, y_hat1))

# #exemple 3
# x2 = np.array([0, 15, -9, 7, 12, 3, -21]).reshape(-1, 1)
# theta2 = np.array([[0.], [1.]]).reshape(-1, 1)
# y_hat2 = predict_(x2, theta2)
# y2 = np.array([2, 14, -13, 5, 12, 4, -19]).reshape(-1, 1)
# print(loss_(y2, y_hat2))

# #exemple 4
# print(loss_(y2, y2))