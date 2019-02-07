import pandas as pd
import numpy as np
from chariot.transformer.formatter.base import BaseFormatter
from chariot.preprocessor import Preprocessor


class Padding(BaseFormatter):

    def __init__(self, padding=0, length=-1,
                 begin_of_sequence=False, end_of_sequence=False):
        super().__init__()
        self.padding = padding
        self.length = length
        self.begin_of_sequence = begin_of_sequence
        self._begin_of_sequence = -1
        self.end_of_sequence = end_of_sequence
        self._end_of_sequence = -1

    def transfer_setting(self, vocabulary_or_preprocessor):
        vocabulary = vocabulary_or_preprocessor
        if isinstance(vocabulary_or_preprocessor, Preprocessor):
            vocabulary = vocabulary_or_preprocessor.vocabulary

        self.padding = vocabulary.pad
        if self.begin_of_sequence:
            self._begin_of_sequence = vocabulary.bos
        if self.end_of_sequence:
            self._end_of_sequence = vocabulary.eos

    def transform(self, column):
        def adjust(sequence, length):
            if self.begin_of_sequence:
                sequence = [self._begin_of_sequence] + sequence
            if self.end_of_sequence > 0:
                sequence = sequence + [self._end_of_sequence]

            if self.length > 0:
                if len(sequence) < self.length:
                    pad_size = self.length - len(sequence)
                    sequence = sequence + [self.padding] * pad_size
                elif len(sequence) > self.length:
                    sequence = sequence[:length]

            return np.array(sequence)

        length = self.length
        if length < 0:
            length = max([len(seq) for seq in column])

        if isinstance(column, pd.Series):
            return column.apply(lambda x: adjust(x, length))
        else:
            return np.array([adjust(x, length) for x in column])

    def inverse_transform(self, column):
        def inverse(sequence):
            excludes = [self.padding,
                        self._begin_of_sequence, self._end_of_sequence]
            inversed = [t for t in sequence if t not in excludes]
            return inversed

        if isinstance(column, pd.Series):
            return column.apply(lambda x: inverse(x))
        else:
            return np.array([inverse(x) for x in column])
