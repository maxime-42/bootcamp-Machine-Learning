import  numpy as np
import pandas as pd
import sys
import os
from day02.ex09.data_spliter import data_spliter
from day04.ex08.my_logistic_regression import MyLogisticRegression as Mlr
from utils.utils_function import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
# from day03.ex07.mono_log import  binarize
from day04.ex00.polynomial_model_extended import add_polynomial_features
import pickle
from day03.ex08.other_metrics import f1_score_

def load_model(filename:str):
    path = os.path.join(os.path.dirname(__file__), f"{filename}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


try :

    feature = pd.read_csv('solar_system_census.csv', index_col=0).to_numpy()
    label = pd.read_csv('solar_system_census_planets.csv', index_col=0).to_numpy()
except Exception as error_msg:
    print("Unexpexted error during open file", file=sys.stderr)
else :
    x = add_polynomial_features(minimize(feature), 3)
    x_train, x_test, y_train, y_test = data_spliter(x, label, 0.2)
    f1s = []
    theta = np.random.rand(x.shape[1] + 1, 1)

    for i, lambda_ in enumerate([0., 0.1, 0.3, 0.5, 0.7, 1.], start=1):
        # model = one_vs_all(lambda_, label, x, i)
        filename = f"model_p{i}_l{lambda_}.pkl"
        model = load_model(filename)
        # f1(y_hat, y, label)
        pred = model.predict_(x_train)
        print(f"Lambda {lambda_} f1:  { f1_score_(y_train, pred)}")
