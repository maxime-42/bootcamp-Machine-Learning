import  numpy as np
import pandas as pd
import sys
from day02.ex09.data_spliter import data_spliter
from day04.ex08.my_logistic_regression import MyLogisticRegression as Mlr
from utils.utils_function import minimize

from day03.ex07.mono_log import  binarize
from day04.ex00.polynomial_model_extended import add_polynomial_features
import pickle
import os
import sys
from day03.ex08.other_metrics import f1_score_



def model_save(model, poly, lambd):
    path = os.path.join(os.path.dirname(__file__), f"model_p{poly}_l{lambd}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def get_accuracy(mlr, x, y):
    y_hat = mlr.predict_(x)
    
    y_hat_binary = binarize(y_hat, 0.5, lambda a, b: np.where(a >= b))
    y_binary = binarize(y, 0.5, lambda a, b: np.where(a >= b))

    # good = np.sum(Y_hat_binary == y_binary)
    # total = Y.shape[0]
    # return good / total
    return f1_score_(y_binary, y_hat_binary)
    # return f1_score_(y, Y_hat_binary)



def one_vs_all(lamda_current, y, x, i):
    theta = np.random.rand(x.shape[1] + 1, 1)


    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.2)

    alpha = 1e-3
    model = Mlr(theta, alpha, max_iter=20000, lambda_=lamda_current)

    model.fit_(x_train,  y_train)

    print(f"Lambda {lambda_} f1:  { get_accuracy(model, x_test, y_test)}")

    return model

if __name__ == "__main__":
    try :
        feature = pd.read_csv('solar_system_census.csv', index_col=0).to_numpy()
        label = pd.read_csv('solar_system_census_planets.csv', index_col=0).to_numpy()
    except Exception as error_msg:
        print("Unexpexted error during open file", file=sys.stderr)
    else :
        x = add_polynomial_features(minimize(feature), 3)
        for i, lambda_ in enumerate([0., 0.1, 0.3, 0.5, 0.7, 1.], start=1):
            model = one_vs_all(lambda_, label, x, i)
            model_save(model, i, lambda_)   