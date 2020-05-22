import numpy as np
from scipy.signal import argrelmax
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

RANDOM_STATE = 7
rng = np.random.RandomState(RANDOM_STATE)


class Detector(BaseEstimator):
    def __init__(self, threshold=0.5, order=40):
        self.threshold = threshold
        self.order = order
        self.step_template = None

    def fit(self, X, y):
        assert len(X) == len(
            y), f"Wrong dimensions (len(X): {len(X)}, len(y): {len(y)})."

        # take a step at random
        index = rng.choice(len(X))
        sensor_data, step_list = X[index], y[index]
        step_index = rng.choice(len(step_list))
        start, end = step_list[step_index]
        self.step_template = sensor_data.signal[[
            "AX", "AY", "AZ", "AV"]][start:end]
        return self

    def predict(self, X):
        y_pred = list()
        for sensor_data in X:
            score = (sensor_data.signal[["AX", "AY", "AZ", "AV"]]  # select the accelerations
                     # sliding window, same shape as the template
                     .rolling(self.step_template.shape[0], center=True)
                     # correlations
                     .apply(lambda x: np.diag(np.corrcoef(x, self.step_template, rowvar=False), k=4))
                     # max pooling
                     .max(axis=1)
                     # to remove NaNs at the edges
                     .fillna(0))
            # set to 0 all values below a threshold
            score[score < self.threshold] = 0.0
            # to find local maxima
            indexes, = argrelmax(score.to_numpy(), order=self.order)
            predicted_steps = [[t - self.order, t + self.order]
                               for t in indexes]
            y_pred += [predicted_steps]
        return np.array(y_pred, dtype=list)


def get_estimator():
    # step detection
    detector = Detector()

    # make pipeline
    pipeline = Pipeline(steps=[('detector', detector)])

    return pipeline
