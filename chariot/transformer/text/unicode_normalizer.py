import unicodedata
from chariot.transformer.text.base import TextNormalizer


class UnicodeNormalizer(TextNormalizer):

    def __init__(self, form="NFKC", copy=True):
        super().__init__(copy)
        self.form = form

    def apply(self, text):
        return unicodedata.normalize(self.form, text)
