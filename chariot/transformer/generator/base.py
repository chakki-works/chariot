from sklearn.base import BaseEstimator, TransformerMixin


class BaseGenerator(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, index, length):
        raise Exception("You should implements transform in subclass.")

    def generate(self, data, index, length):
        raise Exception("You should implements generate in subclass.")
