import numpy as np
from chariot.transformer.generator.base import BaseGenerator


class SourceGenerator(BaseGenerator):

    def __init__(self):
        super().__init__()

    def generate(self, data, index, length):
        _to = data[index:index+length]
        _from = self.transform(data, index, length)
        return _from, _to


class ShuffledSource(SourceGenerator):

    def __init__(self):
        super().__init__()

    def transform(self, data, index, length):
        original = data[index:index+length]
        _to = np.apply_along_axis(np.random.permutation, 1, original)
        return _to
