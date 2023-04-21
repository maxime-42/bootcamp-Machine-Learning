import numpy as np
import sys

def verify_type(y: np.ndarray, y_hat: np.ndarray):
    """ Checking the type of y and y_hat and the data_type of the target
    vector and predicted vector.
    Args:
        y    : [np.ndarray] target vector
        y_hat: [np.ndarray] predicted target vector.
    Return:
        None
    Remark:
        if there are some unconsistency, print message error then  program stopped
    """
    if (not isinstance(y, np.ndarray))   or (not isinstance(y_hat, np.ndarray)):
        print("Unexpected type for y or y_hat.")
        sys.exit()
    if y.dtype.kind != y_hat.dtype.kind:
        print("Unmatching data type.")
        sys.exit()


def verify_shape(y: np.ndarray, y_hat: np.ndarray):
    """ Checking the shape of the target and prediction vectors
        check : data lenght and dimenion if there are some unconsistency program stopped
    Args:
        y    : np.ndarray lable vector
        y_hat: np.ndarray predicted  vector.
    Return:
        None
    Remark:
        if there are some unconsistency, the program stopped
    """
    if y.shape[0] != y_hat.shape[0]:
        print("Error length between y and y_hat.")
        sys.exit()
    if (y.ndim > 2) or (y.ndim > 2) or (y.ndim != y_hat.ndim):
        print("Error dimension between y and y_hat.")
        sys.exit()
    
def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
        Raises:
        This function should not raise any Exception.
    """
    verify_type(y, y_hat)
    verify_shape(y, y_hat)
    return np.sum(y_hat == y)/len(y_hat)

def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
        y:a numpy.ndarray for the correct labels
            y_hat:a numpy.ndarray for the predicted labels
        pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The precision score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    verify_type(y, y_hat)
    verify_shape(y, y_hat)
    TP = np.sum((y == pos_label) & (y_hat == pos_label))
    FP = np.sum((y != pos_label) & (y_hat == pos_label))
    return TP / (TP + FP)




def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
        y:a numpy.ndarray for the correct labels
        y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
        The recall score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    verify_type(y, y_hat)
    verify_shape(y, y_hat)
    TP = ((y == pos_label) & (y_hat == pos_label)).sum()
    FN = ((y == pos_label) & (y_hat != pos_label)).sum()

    recall = TP / (TP + FN) 

    return recall


def f1_score_(y: np.ndarray, y_hat: np.ndarray, pos_label=1):
    """
    Compute the f1 score.
    Args:
        y:a [numpy.array] for the correct labels
        y_hat: [numpy.array] for the predicted labels
        pos_label: [str, int], class on which f1_score is reported (default=1)
    Return:
        The f1 score as a float.
        None if any error.
    Raises:
        This function should not raise any Exception.
    """
    verify_type(y, y_hat)
    verify_shape(y, y_hat)
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    if (precision == 0) & (recall == 0):
        return  0.0
    return 2 * precision * recall / (precision + recall)
