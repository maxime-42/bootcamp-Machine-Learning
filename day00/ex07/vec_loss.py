"""import """
import numpy as np

# import bootcamp_ml.ex06 as  test_loss 
# from bootcamp_ml.ex06 import loss_ as test_loss
from day00.ex06.loss import loss_  as cost_function

def loss_(y:np.ndarray, y_hat:np.ndarray):
    """
        Computes the half mean squared error of two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.
    Args:
        y: has to be an numpy.array, a vector.
        y_hat: has to be an numpy.array, a vector.
    Returns:
        The half mean squared error of the two vectors as a float.
        None if y or y_hat are empty numpy.array.
        None if y and y_hat does not share the same dimensions. 
    Raises:
        This function should not raise any Exceptions.
    """
    return cost_function(y, y_hat)

X = np.array([[0], [15], [-9], [7], [12], [3], [-21]])
Y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])
# Example 1:
print(loss_(X, Y))
# Output:
#2.142857142857143
# Example 2:
print(loss_(X, X))
# Output:
