from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer


def _demean(X_list, copy=True):
    """
    Demean each signal in the list X_list.
    In general, be careful not to change the order of X_list.
    """

    X_transformed = list()

    for walk in X_list:
        # Do not forget that walk is (probably) not an array but an instance
        # of WalkSignal.
        # The "real" signal is walk.signal
        if copy:
            walk_transformed = deepcopy(walk)
            walk_transformed.signal -= walk_transformed.signal.mean(axis=0)
            X_transformed.append(walk_transformed)
        else:
            walk.signal -= walk.signal.mean(axis=0)
            X_transformed.append(walk)

    return X_transformed


class Detector(BaseEstimator):
    """Dummy detector which ouputs 1-second long steps every two seconds."""
    def fit(self, X, y):
        assert len(X) == len(
            y), f"Wrong dimensions (len(X): {len(X)}, len(y): {len(y)})."

        return self

    def predict(self, X):
        y_pred = list()
        for signal in X:
            n_sample = signal.signal.shape[0]
            # if signal is less than 1s long, no steps within
            if n_sample < 100:
                step_list = list()
            else:
                step_list = [[mid - 50, mid + 50 - 1]
                             for mid in range(50, n_sample - 50, 200)]
            y_pred += [step_list]

        return np.array(y_pred, dtype=list)


def get_estimator():

    # preprocessing
    demean_transformer = FunctionTransformer(_demean,
                                             validate=False,
                                             kw_args={"copy": False})

    # step detection on preprocessed data
    detector = Detector()

    # make pipeline
    pipeline = Pipeline(steps=[('demean', demean_transformer),
                               ('detector', detector)])

    return pipeline
