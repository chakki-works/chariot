from chariot.transformer.base_preprocessor import BasePreprocessor
from .ja_tokenizer import MeCabTokenizer
from .ja_tokenizer import JanomeTokenizer
from .spacy_tokenizer import SpacyTokenizer


class Tokenizer(BasePreprocessor):

    def __init__(self, lang="en"):
        self.lang = lang
        self._tokenizer = None
        self.set_tokenizer()

    def set_tokenizer(self):
        if self.lang == "ja":
            try:
                self.tokenizer = MeCabTokenizer()
            except Exception as ex:
                self.tokenizer = JanomeTokenizer()
        else:
            self.tokenizer = SpacyTokenizer(self.lang)

    def apply(self, text):
        return self.tokenizer.tokenize(text)

    def __reduce_ex__(self, proto):
        return type(self), (self.lang,)
