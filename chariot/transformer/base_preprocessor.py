from sklearn.base import BaseEstimator, TransformerMixin


class BasePreprocessor(BaseEstimator, TransformerMixin):

    def apply(self, elements):
        raise Exception("You have to implements apply")

    def fit(self, X, y=None):
        return self

    def _transform(self, X, func):
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
