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
    data = np.hstack((x, y))

    np.random.shuffle(data)
    # print(data)
    index = int(proportion * x.shape[0])
	#slicing:
	#  :-1 means to select all columns except the last one, 
	#  -1: means to select only the last column.

	#les `index` premiere element
    x_train, y_train = data[:index, :-1], data[:index, -1:]
    #à partir de 'index' element 
    x_test, y_test = data[index:, :-1], data[index:, -1:]
    return (x_train, x_test, y_train, y_test)