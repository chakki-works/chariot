import numpy as np
from chariot.transformer.adjuster.base import BaseAdjuster


class CategoricalLabel(BaseAdjuster):

    def __init__(self, class_count=-1):
        super().__init__()
        self.class_count = class_count

    def transform(self, column):
        y = np.array(column, dtype="int")
        input_shape = y.shape
        y = y.ravel()
        n = y.shape[0]
        categorical = np.zeros((n, self.class_count), dtype=np.float32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (self.class_count,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
