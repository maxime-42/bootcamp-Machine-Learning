import numpy as np

from log_pred import logistic_predict_

x = np.array([4]).reshape((-1, 1))
theta = np.array([[2], [0.5]])

print("Exemple 1:")
print(logistic_predict_(x, theta))

print("\nExemple 2:")
x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
print(logistic_predict_(x2, theta))

print("\nExemple 3:")
x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])

print(logistic_predict_(x3, theta3))