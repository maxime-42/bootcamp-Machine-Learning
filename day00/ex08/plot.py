"""import utils module"""
import numpy as np
import matplotlib.pyplot as plt

from day00.ex02.prediction import simple_predict 

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
    print("model= ", model)
    if model is None:
        return None
    plt.plot(x, model, color='r')
    # print(f"x = {x}")
    # print(f"model = {model}")
    # print(f"y = {y}")

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
