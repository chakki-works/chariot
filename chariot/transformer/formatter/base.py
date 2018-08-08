from sklearn.base import BaseEstimator, TransformerMixin


class BaseFormatter(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, column):
        raise Exception("You have to implements transform method.")
