import copy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, copy=True):
        self.copy = copy

    def apply(self, elements):
        raise Exception("You have to implements apply")

    def fit(self, X, y=None):
        return self

    def _transform(self, X, func):
        X = self.check_array(X)
        if isinstance(X, dict):
            for k in X:
                X[k] = [func(row) for row in X[k]]
            return X
        elif isinstance(X, (list, tuple)):
            for i in range(len(X)):
                row = X[i]
                if len(row) > 0 and isinstance(row[0], (list, tuple)):
                    # row has multiple columns
                    _row = []
                    for column in row:
                        _row.append(func(column))
                    X[i] = _row
                else:
                    X[i] = func(row)
        return X

    def transform(self, X):
        return self._transform(X, self.apply)

    def check_array(self, X):
        def array_copy(array):
            copied = True
            array = array
            if isinstance(array, (np.ndarray, np.generic)):
                array = np.array(array)
            elif isinstance(array, list):
                array = list(array)
            elif isinstance(array, tuple):
                array = tuple(array)
            else:
                copied = False
            return copied, array

        if not self.copy:
            return X
        else:
            copied, array = array_copy(X)
            if copied:
                return array
            else:
                if isinstance(X, dict):
                    _X = {}
                    for k in X:
                        copied, array = array_copy(X[k])
                        _X[k] = array
                        if not copied:
                            _X[k] = X[k]
                    return _X
                else:
                    return copy.deepcopy(X)
