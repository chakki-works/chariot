import numpy as np


class Adjuster():

    def __init__(self, vocab_size=0, padding=-1, unknown=-1,
                 begin_of_seq=-1, end_of_seq=-1):
        self.vocab_size = vocab_size
        self.padding = padding
        self.unknown = unknown
        self.begin_of_seq = begin_of_seq
        self.end_of_seq = end_of_seq

    def adjust(self, indices, padding=-1, bos=False, eos=False):
        if bos:
            indices = [self.begin_of_seq] + indices
        if eos:
            indices = indices + [self.end_of_seq]

        if padding > 0:
            pad = self.padding
            if len(indices) < padding:
                pad_size = padding - len(indices)
                indices = indices + [pad] * pad_size
            elif len(indices) > padding:
                indices = indices[:padding]

        return indices

    def to_categorical(self, array):
        y = np.array(array, dtype="int")
        input_shape = y.shape
        y = y.ravel()
        n = y.shape[0]
        categorical = np.zeros((n, self.vocab_size), dtype=np.float32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (self.vocab_size,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
