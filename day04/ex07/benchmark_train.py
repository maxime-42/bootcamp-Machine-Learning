import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys

path = os.path.join(os.path.dirname(__file__), '..', 'ex06')
sys.path.insert(1, path)
from day04.ex06.ridge import MyRidge as MyLR
from day04.ex00.polynomial_model_extended import add_polynomial_features
from utils.utils_function import  minimize
from day02.ex09.data_spliter import  data_spliter 


def save_model(data):
    path = os.path.join(os.path.dirname(__file__), f"model.pkl")
    with open(path, 'wb') as f:
        pickle.dump(data, f)
        


def drawn_scatter(X, pred_price, true_price, costs):
    plot_dim = 2
    fig, axs_ = plt.subplots(plot_dim, plot_dim)
    axs = []
    for sublist in axs_:
        for item in sublist:
            axs.append(item)

    for idx_feature, feature in enumerate(X.T):
        for idx_pred, y_hat in enumerate(pred_price):
            c = ['r', 'y', 'm', 'b']
            color = c[idx_pred]
            axs[idx_feature].scatter(feature, y_hat, s=.1, c=color, label=f"Poly {idx_pred}")
        axs[idx_feature].scatter(feature, true_price, s=0.1, c='g', label="True")
        axs[idx_feature].legend()

    legend = [f"Pol {i}" for i in range(1, len(costs) + 1)]
    axs[-1].bar(legend,costs)
    plt.show()

def regularized_factor(poly_del):
    factor = poly_del == 1 if poly_del else poly_del + 1
    return factor / 10.0

def create_model(x, y, poly_degre):
    """
        this functon create a model then train it
        args: 
            x : features (weight,prod_distance,time_delivery)
            y: target it is the prize of avocat 
    """
    x_poly = add_polynomial_features(x, poly_degre)
    x_train, x_test, y_train, y_test = data_spliter(x_poly, y, 0.8)
    
    theta = np.array([0] * (poly_degre * x.shape[1] + 1)).reshape(-1, 1)
    
    lambda__ = regularized_factor(poly_degre) 

    model = MyLR(thetas=theta, alpha= 1e-2, max_iter=100000, lambda_=lambda__)
    
    model.fit_(x_train, y_train)
    
    pred = model.predict_(x_test)
    cost = model.loss_(y_test, pred)
    
    print(f"{cost = }")
    
    return cost, model, model.predict_(x_poly)


if __name__ == "__main__":
    try:

        file = pd.read_csv("space_avocado.csv")

        feature = np.array(file[["weight","prod_distance","time_delivery"]]).reshape(-1,3)
        
        label = np.array(file["target"]).reshape(-1,1)
    except Exception as error_msg:
        print("Error happened during open file", file=sys.stderr)
    else:
        x_minimize = minimize(feature)
        costs = []
        preds = []
        lowest_mse = float('inf')
        best_model = any 
        for i in range(1, 5):
            cost_current,  model_current, pred = create_model(x_minimize, label, i)
            costs.append(cost_current)
            preds.append(pred)
            if (cost_current < lowest_mse ):
                lowest_mse = cost_current
                model = model_current

        save_model(model)
        drawn_scatter(x_minimize, preds, label, costs)
