from chariot.transformer.text.base import TextFilter


class SymbolFilter(TextFilter):

    def __init__(self, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                 split=" ", copy=True):
        super().__init__(copy)
        self.filters = filters
        self.split = split

    def apply(self, text):
        translate_dict = dict((c, self.split) for c in self.filters)
        translate_map = str.maketrans(translate_dict)
        _text = text.translate(translate_map)

        return _text
