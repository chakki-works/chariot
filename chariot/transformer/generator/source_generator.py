from chariot.transformer.generator.base import BaseGenerator


class SourceGenerator(BaseGenerator):

    def __init__(self):
        super().__init__()

    def generate(self, data, index, length):
        _to = data[index:index+length]
        _from = self.transform(data, index, length)
        return _from, _to
