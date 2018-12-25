import pandas as pd
import numpy as np


class TargetGenerator():

    def __init__(self):
        pass

    def generate(self, data, index, length):
        raise Exception("You should implements generate in subclass.")


class ShiftGenerator(TargetGenerator):

    def __init__(self, shift=1):
        super().__init__()
        self.shift = shift

    def generate(self, data, index, length):
        _from = data[index:index+length]
        _to = data[index+self.shift:index+self.shift+length]
        return _from, _to


class ShuffleGenerator(TargetGenerator):

    def __init__(self):
        super().__init__()

    def generate(self, data, index, length):
        _from = data[index:index+length]
        _to = np.apply_along_axis(np.random.permutation, 1, _from)
        return _from, _to
