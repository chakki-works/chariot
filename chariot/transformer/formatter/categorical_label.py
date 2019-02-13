import numpy as np
from chariot.transformer.formatter.base import BaseFormatter
from chariot.preprocessor import Preprocessor


class CategoricalLabel(BaseFormatter):

    def __init__(self, num_class=-1):
        super().__init__()
        self.num_class = num_class

    def transfer_setting(self, vocabulary_or_preprocessor):
        vocabulary = vocabulary_or_preprocessor
        if isinstance(vocabulary_or_preprocessor, Preprocessor):
            vocabulary = vocabulary_or_preprocessor.vocabulary
        self.num_class = vocabulary.count

    def transform(self, column):
        y = np.array(column, dtype="int")
        input_shape = y.shape
        if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
            input_shape = tuple(input_shape[:-1])
        y = y.ravel()
        n = y.shape[0]
        categorical = np.zeros((n, self.num_class), dtype=np.float32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (self.num_class,)
        categorical = np.reshape(categorical, output_shape)
        return categorical

    def inverse_transform(self, batch):
        return np.argmax(batch, axis=1)
