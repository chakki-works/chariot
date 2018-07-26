import pandas as pd
import numpy as np
from chariot.transformer.adjuster.base import BaseAdjuster


class Padding(BaseAdjuster):

    def __init__(self, padding=0, length=-1,
                 begin_of_sequence=-1, end_of_sequence=-1):
        super().__init__()
        self.padding = padding
        self.length = length
        self.begin_of_sequence = begin_of_sequence
        self.end_of_sequence = end_of_sequence

    def transform(self, column):
        def adjust(sequence, length):
            if self.begin_of_sequence > 0:
                sequence = [self.begin_of_sequence] + sequence
            if self.end_of_sequence > 0:
                sequence = sequence + [self.end_of_sequence]

            if self.padding > 0:
                if len(sequence) < length:
                    pad_size = length - len(sequence)
                    sequence = sequence + [self.padding] * pad_size
                elif len(sequence) > length:
                    sequence = sequence[:length]

            return np.array(sequence)

        length = self.length
        if length < 0:
            length = max([len(seq) for seq in column])

        if isinstance(column, pd.Series):
            return column.apply(lambda x: adjust(x, length))
        else:
            return np.array([adjust(x, length) for x in column])
