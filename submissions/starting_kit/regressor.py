import numpy as np
from sklearn.base import BaseEstimator


class Regressor(BaseEstimator):

    def fit(self, X, y):
        assert len(X) == len(
            y), f"Wrong dimensions (len(X): {len(X)}, len(y): {len(y)})."

        return self

    def predict(self, X):
        y_pred = list()
        for signal in X:
            n_sample = signal.signal.shape[0]
            if n_sample < 100:  # if signal is less than 1 sec long, no steps within
                step_list = list()
            else:
                step_list = [[mid-50, mid+50-1]
                             for mid in range(50, n_sample, 100)]
            y_pred += [step_list]

        return np.array(y_pred, dtype=list)
