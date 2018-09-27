from chariot.transformer.base_preprocessor import BasePreprocessor
from chariot.util import apply_map
from .ja_tokenizer import MeCabTokenizer
from .ja_tokenizer import JanomeTokenizer
from .spacy_tokenizer import SpacyTokenizer
from .split_tokenizer import SplitTokenizer


class Tokenizer(BasePreprocessor):

    def __init__(self, lang="en", copy=True):
        super().__init__(copy)
        self.lang = lang
        self._tokenizer = None
        self.set_tokenizer()

    def set_tokenizer(self):
        if self.lang == "ja":
            try:
                self.tokenizer = MeCabTokenizer()
            except Exception as ex:
                self.tokenizer = JanomeTokenizer()
        elif self.lang is None:
            self.tokenizer = SplitTokenizer()
        else:
            self.tokenizer = SpacyTokenizer(self.lang)

    def transform(self, X):
        _X = apply_map(X, self.tokenizer.tokenize, self.copy)
        return _X

    def __reduce_ex__(self, proto):
        return type(self), (self.lang, self.copy)
