import numpy as np

def loss_(y:np.ndarray, y_hat:np.ndarray):
    """
        Computes the mean squared error of two non-empty numpy.array, without any for loop.
        The two arrays must have the same dimensions.
        Args:
            y: has to be an numpy.array, a vector.
            y_hat: has to be an numpy.array, a vector.
        Return:
            The mean squared error of the two vectors as a float.
            None if y or y_hat are empty numpy.array.
            None if y and y_hat does not share the same dimensions.
            None if y or y_hat is not of expected type.
        Raises:
            This function should not raise any Exception.
    """
    if all([isinstance(obj, np.ndarray) and len(obj) > 1 for obj in  (y, y_hat)]) is False or y.shape != y_hat.shape:
        return None
    squared_distances = (y_hat - y)**2
    divizer = 2 * len(y_hat)
    cost = 1/(divizer) * np.sum(squared_distances)
 
    print(f"{cost}\n")
    return cost