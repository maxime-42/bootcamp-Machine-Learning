import numpy as np


from  day01.ex01.vec_gradient  import gradient
from  day01.ex00.gradient  import  check_requirement_args


def fit_(x:np.ndarray, y:np.ndarray, theta:np.ndarray, alpha:float, max_iter:float):
    """
    Description:
        Fits the model to the training dataset contained in x and y.
    Args:
    x: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
    y: has to be a numpy.ndarray, a vector of dimension m * 1: (number of training examples, 1).
        theta: has to be a numpy.ndarray, a vector of dimension 2 * 1.
        alpha: has to be a float, the learning rate
        max_iter: has to be an int, the number of iterations done during the gradient descent
    Returns:
        new_theta: numpy.ndarray, a vector of dimension 2 * 1.
        None if there is a matching dimension problem.
    Raises:
        This function should not raise any Exception.
    """
    if check_requirement_args(x, y, theta) is False: 
        return None
    derive = np.ones(2)
    for i in range(max_iter):
        derive = gradient(x, y, theta)
        theta = theta - alpha * derive
    return theta

# x = np.array([[12.4956442], [21.5007972], [31.5527382], [48.9145838], [57.5088733]])
# y = np.array([[37.4013816], [36.1473236], [45.7655287], [46.6793434], [59.5585554]])
# theta= np.array([1, 1]).reshape((-1, 1))

# Example 0:
# theta1 = fit_(x, y, theta, alpha=5e-8, max_iter=1500000)
# print(theta1)
# Output:
# array([[1.40709365],
# [1.1150909 ]])
# # Example 1:
# predict(x, theta1)
# # Output:
# array([[15.3408728 ],
# [25.38243697],
# [36.59126492],
# [55.95130097],
# [65.53471499]])
# • You can create more training data by generating an x array
# with random values and computing the corresponding y vector as
# a linear expression of x. You can then fit a model on this
# artificial data and find out if it comes out with the same θ
# coefficients that first you used.
# • It is possible that θ0 and θ1 become "nan". In that case, it
# means you probably used a learning rate that is too large.
# 15
