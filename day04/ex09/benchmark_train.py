import  numpy as np
import pandas as pd
import sys
from day02.ex09.data_spliter import data_spliter
from day04.ex08.my_logistic_regression import MyLogisticRegression as Mlr
from utils.utils_function import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
# from day03.ex07.mono_log import  binarize
from day04.ex00.polynomial_model_extended import add_polynomial_features
import pickle

import os
import sys
from day03.ex08.other_metrics import f1_score_



def get_bin_Y(Y, feature):
    Y_ = Y.copy()
    Y_[(Y == feature)] = 1.
    Y_[(Y != feature)] = 0.
    return Y_

def model_save(model, poly, lambd):
    path = os.path.join(os.path.dirname(__file__), f"model_p{poly}_l{lambd}.pkl")
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def one_vs_all(lamda_, y, x, i):
    theta = np.random.rand(x.shape[1] + 1, 1)

    # y_binary = binarize(y, i, lambda a, b: np.where(a == b))
    
    x_train, x_test, y_train, y_test = data_spliter(x, y, 0.2)
    # 
    model = Mlr(theta, 1e-3, max_iter=20000)
    y_train_binary = get_bin_Y(y_train, y)
    model.fit_(x_train,  y_train_binary)
    pred = model.predict_(x_test)
    # print(f"{pred =} \n\n")
    print(f"Lambda {lambda_} f1:  { f1_score_(x_test, y_test)}")

    return model

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