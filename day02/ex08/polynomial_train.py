import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PACE = 0.1

from day02.ex05.mylinearregression import MyLinearRegression as MyLR

from day02.ex07.polynomial_model import add_polynomial_features


def score(scor_plot, power:int, linear_r:MyLR, x:np.ndarray, y:np.ndarray):
    y_hat = linear_r.predict_(x)
    mse:float = linear_r.mse_(y, y_hat)
    print(f"POWER: {power}; {mse = }")
    print(f"Thetas:\n{linear_r.thetas}")
    scor_plot.bar(power,mse)

def model(model_plot, power:int, linear_r:MyLR, y:np.ndarray):
    
    continuous_x =  np.arange(1, 7 + PACE, PACE).reshape(-1, 1)
    x = add_polynomial_features(continuous_x, power)
    y_hat = linear_r.predict_(x)
    # print(y_hat)

    model_plot.plot(continuous_x, y_hat, label="$%d_{th} curve$" %power)


def main():
    data = pd.read_csv("are_blue_pills_magics.csv")
    x = np.array(data["Micrograms"])
    y = np.array(data["Score"])

    params1 = {
        'theta' :np.array([[-20],[ 160]]).reshape(-1,1),
        'max_iter': 100000,
        'alpha': 1e-2,
    }
    params2 = {
        'theta' :np.array([[-20],[ 160],[ -80]]).reshape(-1,1),
        'max_iter': 100000,
        'alpha': 1e-3,
    }
    params3 = {
        'theta' :np.array([[-20],[ 160],[ -80],[ 10]]).reshape(-1,1),
        'max_iter': 100000,
        'alpha': 1e-5,
    }
    params4 = {
        'theta' :np.array([[-20],[ 160],[ -80],[ 10],[ -1]]).reshape(-1,1),
        'max_iter': 1000000,
        'alpha': 1e-7,
    }

    params5 = {
        'theta': np.array([[1140],[ -1850],[ 1110],[ -305],[ 40],[ -2]]).reshape(-1,1), 
        'max_iter': 1000000,
        'alpha': 1e-10,
    }

    params6 = {
        'theta' : np.array([[9110],[ -18015],[ 13400],[ -4935],[ 966],[ -96.4],[ 3.86]]).reshape(-1,1),
        'max_iter': 1000000,
        'alpha': 1e-10,
    }

    score_ax = plt.subplot()
    score_ax.set_title("MSE in function of polynomial's degree")
    score_ax.set_xlabel("x degree")
    score_ax.set_ylabel("MSE")
    score_ax.grid()

    model_ax = plt.subplot()
    model_ax.set_title("Score depending on taken blue pills (Micrograms)")
    model_ax.set_xlabel("Micrograms")
    model_ax.set_ylabel("Score")

    model_ax.plot(x, y, "o", label="$S_{true}$")

    # print(x)
    for i, param in enumerate([params1, params2, params3, params4, params5,params6 ] ,start=1):
        new_x = add_polynomial_features(x, i)
        print(f"Training #{i} model...")

        # print(new_x)
        # print(param)
        linear_r = MyLR(param['theta'], alpha=param['alpha'], max_iter=param['max_iter'])
        # print(linear_r.predict_(new_x))
        # print(f"Training #{i} model...")
        linear_r.fit_(new_x, y)
        # score(score_ax, i, linear_r, new_x, y)
        print(linear_r.thetas)

        # model(model_ax, i, linear_r, y)

    score_ax.legend()
    model_ax.legend()
    plt.show()

if __name__ == "__main__":
    main()