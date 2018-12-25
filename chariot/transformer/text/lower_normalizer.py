from chariot.transformer.text.base import TextNormalizer


class LowerNormalizer(TextNormalizer):

    def __init__(self, copy=True):
        super().__init__(copy)

    def apply(self, text):
        return text.lower()
