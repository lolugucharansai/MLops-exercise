import numpy as np
from Initial_job.main2 import train_model

def test_train_model():
    X_train = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    X_test = np.array([7, 8, 9, 10, 11, 12]).reshape(-1, 1)
    y_train = np.array([10, 9, 8, 8, 6, 5])
    y_test = np.array([5, 6, 7, 7, 9, 10])
    reg_rate = 1.2

    reg_model = train_model(reg_rate, X_train, X_test, y_train, y_test)

    preds = reg_model.predict([[1], [2]])
    np.testing.assert_almost_equal(preds, [8, 8]) 