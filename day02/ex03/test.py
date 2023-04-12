import numpy as np
from gradient  import gradient


if __name__ == "__main__":
    x = np.array([[-6, -7, -9],
                  [13, -2, 14],
                  [-7, 14, -1],
                  [-8, -4, 6],
                  [-5, -9, 6],
                  [1, -5, 11],
                  [9, -11, 8]])
    y = np.array([[2], [14], [-13], [5], [12], [4], [-19]])

    print("# Example 0:")
    theta1 = np.array([[0], [3], [0.5], [-6]])
    grad = gradient(x, y, theta1)
    # Output:
    expected_grad = np.array([[-33.71428571],
                              [-37.35714286],
                              [183.14285714],
                              [-393.]])
    print("my gradient:\n", grad)
    print("expected gradient:\n", expected_grad)

    print("\n# Example 1:")
    theta2 = np.array([[0], [0], [0], [0]])
    grad = gradient(x, y, theta2)
    # Output:
    expected_grad = np.array([[-0.71428571],
                              [0.85714286],
                              [23.28571429],
                              [-26.42857143]])
    print("my gradient:\n", grad)
    print("expected gradient:\n", expected_grad)
