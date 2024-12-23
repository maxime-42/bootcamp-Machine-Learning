import  numpy as np
import pandas as pd
import sys
import os
from day02.ex09.data_spliter import data_spliter
# from day04.ex08.my_logistic_regression import MyLogisticRegression as Mlr
from utils.utils_function import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
# from day03.ex07.mono_log import  binarize
from day04.ex00.polynomial_model_extended import add_polynomial_features
import pickle
from benchmark_train import get_accuracy
import seaborn as sns
sns.set()

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
    
    lambdas = [0., 0.1, 0.3, 0.5, 0.7, 1.]
    score_lst = []
    for i, lambda_ in enumerate(lambdas, start=1):
        # model = one_vs_all(lambda_, label, x, i)
        filename = f"model_p{i}_l{lambda_}.pkl"
        model = load_model(filename)
        pred = model.predict_(x_train)
        score = get_accuracy(model, x_test, y_test)
        print(f"Poly {i}: Lambda {lambda_} f1:  {score}")
        score_lst.append(score)
    
    sns.barplot(x=lambdas, y=score_lst)
    plt.show()