from copy import deepcopy


class FeatureExtractor:

    def fit(self, X, y):
        return self

    def transform(self, X_list, copy=True):
        """
        Demean each signal in the list X_list.

        In general, be careful not to change the order of X_list.
        """

        X_transformed = list()

        for walk in X_list:
            # Do not forget that walk is (probably) not an array but an instance of WalkSignal.
            # The "real" signal is walk.signal
            if copy:
                walk_transformed = deepcopy(walk)
                walk_transformed.signal -= walk_transformed.signal.mean(axis=0)
                X_transformed.append(walk_transformed)
            else:
                walk.signal -= walk.signal.mean(axis=0)
                X_transformed.append(walk)

        return X_transformed
