import numpy as np
from numpy import random

def data_spliter(x, y, proportion):
    """
    Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
        x: has to be an numpy.array, a matrix of dimension m * n.
        y: has to be an numpy.array, a vector of dimension m * 1.
        proportion: has to be a float, the proportion of the dataset that will be assigned to the
        training set.
    Return:
        (x_train, x_test, y_train, y_test) as a tuple of numpy.array
        None if x or y is an empty numpy.array.
        None if x and y do not share compatible dimensions.
        None if x, y or proportion is not of expected type.
    Raises:
        This function should not raise any Exception.
    """
    # Determine the number of training samples based on the given proportion
    # data = np.concatenate(x, y, axis=0)
    data = np.random.shuffle(np.concatenate(x, y, axis=1))
    index = int(proportion * x.shape[0])

    indices = np.random(x.shape[0])

    x_train, y_train = data[:index, :], data[:index, :]
    x_test, y_test = data[index:, :-1], data[index:, -1:]
    return (x_train, x_test, y_train, y_test)