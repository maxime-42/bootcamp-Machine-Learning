import numpy as np
import pandas as pd


def confusion_matrix_(y_true, y_hat, labels=None, df_option=True):
    """
    Compute confusion matrix to evaluate the accuracy of a classification.
    Args:
        y:a numpy.array for the correct labels
        y_hat:a numpy.array for the predicted labels
        labels: optional, a list of labels to index the matrix.
        This may be used to reorder or select a subset of labels. (default=None)
        df_option: optional, if set to True the function will return a pandas DataFrame
        instead of a numpy array. (default=False)
    Return:
        The confusion matrix as a numpy array or a pandas DataFrame according to df_option value.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    concat = y_true
    if labels is None:
        concat = np.hstack((y_true, y_hat))
        labels = np.unique(concat)
    else:
        labels = labels

    data = np.zeros((len(labels), len(labels)))
    matrix = pd.DataFrame(data, index=labels, columns=labels, dtype=int)

    for index in labels:
        mask = index == y_true
        for col in labels:
            tmp = col == y_hat
            count = np.sum(tmp & mask)
            matrix[col][index] = count
    if df_option == True :
        return matrix
    return matrix.values