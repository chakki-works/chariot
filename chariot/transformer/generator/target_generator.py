import numpy as np
from chariot.transformer.generator.base import BaseGenerator


class TargetGenerator(BaseGenerator):

    def __init__(self):
        super().__init__()

    def generate(self, data, index, length):
        _from = data[index:index+length]
        _to = self.transform(data, index, length)
        return _from, _to


class ShiftedTarget(TargetGenerator):

    def __init__(self, shift=1):
        super().__init__()
        self.shift = shift

    def transform(self, data, index, length):
        _to = data[index+self.shift:index+self.shift+length]
        return _to


class ShuffledTarget(TargetGenerator):

    def __init__(self):
        super().__init__()

    def transform(self, data, index, length):
        original = data[index:index+length]
        _to = np.apply_along_axis(np.random.permutation, 1, original)
        return _to
