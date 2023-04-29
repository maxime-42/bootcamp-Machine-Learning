import numpy as np
from my_logistic_regression import MyLogisticRegression 
if __name__ == "__main__":
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]])
    Y = np.array([[1], [0], [1]])

    MyLR = MyLogisticRegression
    # mylr = MyLR([2, 0.5, 7.1, -4.3, 2.09], penalty=None)
    mylr = MyLR(np.array([2, 0.5, 7.1, -4.3, 2.09]).reshape(-1, 1))

    print("# Example 0:")
    print(f"{mylr.predict_(X) = }")
    print()

    print("# Example 1:")
    print(f"{mylr.loss_(X,Y) = }")
    print()

    print("# Example 2:")
    print(f"{mylr.fit_(X, Y) = }")
    print(f"{mylr.theta = }")
    print()

    print("# Example 3:")
    print(f"{mylr.predict_(X) = }")
    print()

    print("# Example 4:")
    print(f"{mylr.loss_(X,Y) = }")
    print()













