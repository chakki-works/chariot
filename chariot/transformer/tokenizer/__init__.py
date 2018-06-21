from chariot.transformer.base_preprocessor import BasePreprocessor
from .ja_tokenizer import MeCabTokenizer
from .ja_tokenizer import JanomeTokenizer
from .spacy_tokenizer import SpacyTokenizer


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
        else:
            self.tokenizer = SpacyTokenizer(self.lang)

    def _transform(self, X, func):
        X = self.check_array(X)
        if isinstance(X, dict):
            for k in X:
                X[k] = func(X[k])
            return X
        elif isinstance(X, (list, tuple)):
            if len(X) > 0 and len(X[0]) > 0 and \
               isinstance(X[0][0], (list, tuple)):
                # row has multiple columns
                for c in range(len(X[0])):
                    column = []
                    for r in range(len(X)):
                        column.append(X[r][c])
                    result = func(column)
                    for r in range(len(X)):
                        X[r][c] = result[r]
            else:
                X = func(X)
        return X

    def transform(self, X):
        return self._transform(X, self.tokenizer.apply)

    def __reduce_ex__(self, proto):
        return type(self), (self.lang, self.copy)
