
from fit import fit_
from day02.ex01.prediction import predict_
import numpy as np

if __name__ == "__main__":
    x = np.array([[0.2, 2., 20.],
                  [0.4, 4., 40.],
                  [0.6, 6., 60.],
                  [0.8, 8., 80.]])
    y = np.array([[19.6], [-2.8], [-25.2], [-47.6]])
    theta = np.array([[42.], [1.], [1.], [1.]])
    print("# Example 0:")
    nw_theta = fit_(x, y, theta, alpha=0.0005, max_iter=42000)
    # Output:
    expected_theta = np.array([[41.99], [0.97], [0.77], [-1.20]])
    print("initial value of theta:".ljust(30), theta.reshape(1, -1))
    print("After training value of theta:".ljust(30), nw_theta.reshape(1, -1))
    print("Expected approximate value:".ljust(30),
          expected_theta.reshape(1, -1))

    print("\n# Example 1:")
    pred = predict_(x, nw_theta)
    # Output:
    # np.array([[19.5992..], [-2.8003..], [-25.1999..], [-47.5996..]])
    expected_pred = np.array([[19.5992], [-2.8003], [-25.1999], [-47.5996]])
    print("my prediction:".ljust(20), pred.reshape(1, -1))
    print("expected prediction:".ljust(20), expected_pred.reshape(1, -1))
