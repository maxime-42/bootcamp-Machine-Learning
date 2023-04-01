"""
Standardization, on the other hand, involves transforming the data so that it has a mean of 0 and a standard deviation of 1. 
This is typically done by subtracting the mean of the feature and dividing by the standard deviation of the feature.
"""
import numpy as np


def zscore(x:np.ndarray):
    """
    Computes the normalized version of a non-empty numpy.ndarray using the z-score standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
        None if x is a non-empty numpy.ndarray or not a numpy.ndarray.
    Raises:
        This function shouldn’t raise any Exception.
    """
    if not all([isinstance(x, np.ndarray),  x.shape in [(x.size, 1)]]):
        return None

    xp = np.zeros(len(x))
    mean = np.sum(x) / x.size
    std_dev = np.std(x)
    
    for i, value  in enumerate(x):
         xp[i] = (value - mean) / std_dev
    return xp
   

print("Example 1")
X = np.array([0, 15, -9, 7, 12, 3, -21])
print(zscore(X))

print("Exemple 2")
Y = np.array([2, 14, -13, 5, 12, 4, -19]).reshape((-1, 1))
print(zscore(Y))