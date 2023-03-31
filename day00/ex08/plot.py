"""import utils module"""
import numpy as np
import matplotlib.pyplot as plt

from loss import loss_  
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

def plot_with_loss(x:np.ndarray, y:np.ndarray, theta:np.ndarray):
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
    model = simple_predict(x, theta)
    if model is None:
        return None
    plt.plot(x, model, color='r')
    print(f"x = {x}")
    print(f"model = {model}")
    print(f"y = {y}")

    plt.scatter(x, y)
    coef = np.polyfit(x, model, 1)
    cost = np.poly1d(coef)
    for i in range(len(x)):
        plt.plot([x[i], x[i]], [y[i], cost(x[i])], color='r', linestyle = 'dashed')
    plt.show()



x = np.arange(1,6)
y = np.array([11.52434424, 10.62589482, 13.14755699, 18.60682298, 14.14329568])
# Example 1:
theta1= np.array([18,-1])
plot_with_loss(x, y, theta1)
# Output:

# # Example 2:
theta2 = np.array([14, 0])
plot_with_loss(x, y, theta2)
# # Output:

# #  Example 3:
theta3 = np.array([12, 0.8])
plot_with_loss(x, y, theta3)
