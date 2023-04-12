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
    # split_index = x[:int(proportion * len(x)) ]
    # random.shuffle(x)
    # random.shuffle(y)

    split_pointx = int(proportion *len(x))
    split_pointy = int(proportion * len(y))

    train_x, test_x = np.split(x, [split_pointx])
    train_y, test_y = np.split(y, [split_pointy])

    # train_set.append(x[:int(proportion * len(x))])
    # train_set.append(y[:int(proportion * len(y))])

    print(f"train set {train_y} {train_x}")
    # print({train_x}")

    # print((x[:int(proportion * len(x))], y[:int(proportion * len(y))] ))
