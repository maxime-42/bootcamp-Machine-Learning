
import pickle
import os
import sys
import pandas as pd
import numpy as np
from utils.utils_function import  minimize
from day04.ex00.polynomial_model_extended import add_polynomial_features
from day02.ex09.data_spliter import  data_spliter 

from benchmark_train import drawn_scatter

def load_model():
    path = os.path.join(os.path.dirname(__file__), f"model.pkl")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

if __name__ == "__main__":
    try:

        data = pd.read_csv("space_avocado.csv")

        X = np.array(data[["weight","prod_distance","time_delivery"]]).reshape(-1,3)
        Y = np.array(data["target"]).reshape(-1,1)


    except Exception as error_msg:
        print("Error happened during open file", file=sys.stderr)
    else:
        X_ = minimize(X)
        costs = []
        preds = []
        X_poly = add_polynomial_features(X_, 4)
        lr = load_model()
        x_train, x_test, y_train, y_test = data_spliter(X_poly, Y, 0.8)
        lr.fit_(x_train, y_train)
        lr.predict_(X_poly)
        # print(type(lr))
        y_hat = lr.predict_(X_poly)
        cost = lr.loss_(Y, y_hat)
        print(f"{cost = }")
        costs.append(cost)
        preds.append(y_hat)

        drawn_scatter(X, preds, Y, costs)