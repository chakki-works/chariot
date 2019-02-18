from sklearn.base import BaseEstimator, TransformerMixin
from chariot.util import apply_map


class BasePreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, copy=True):
        self.copy = copy

    def apply(self, elements):
        raise Exception("You have to implements apply")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return apply_map(X, self.apply, inplace=(not self.copy))
