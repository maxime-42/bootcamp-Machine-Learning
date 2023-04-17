import numpy as np
from my_logistic_regression import MyLogisticRegression as MyLR

X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
Y = np.array([[1], [0], [1]])
thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]])
mylr = MyLR(thetas)

print("# Example 0:")
print(mylr.predict_(X))

print("\n# Example 1:")
# print(Y.shape)
print(mylr.loss_(X,Y))

print("\nExample 2:")
mylr.fit_(X, Y)
print(mylr.theta)


# Example 3:
print("\nExample 3:")

print(mylr.predict_(X))

# Example 4:
print("\nExample 4:")
print(mylr.loss_(X,Y))