import re
import copy
import numpy as np
from sklearn.externals import joblib
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import BaseEstimator, TransformerMixin
from chariot.transformer.text.base import TextNormalizer, TextFilter
from chariot.transformer.token.base import TokenFilter, TokenNormalizer
from chariot.transformer.vocabulary import Vocabulary
from chariot.transformer.tokenizer import Tokenizer


class Preprocessor(_BaseComposition, BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer=None,
                 text_transformers=(), token_transformers=(),
                 vocabulary=None, other_transformers=()):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = Tokenizer(self.tokenizer)
        self.text_transformers = list(text_transformers)
        self.token_transformers = list(token_transformers)
        self.vocabulary = vocabulary
        self.other_transformers = list(other_transformers)

    def stack(self, transformer):
        if isinstance(transformer, Tokenizer):
            self.tokenizer = transformer
        elif isinstance(transformer, (TextNormalizer, TextFilter)):
            self.text_transformers.append(transformer)
        elif isinstance(transformer, (TokenFilter, TokenNormalizer)):
            self.token_transformers.append(transformer)
        elif isinstance(transformer, Vocabulary):
            self.vocabulary = transformer
        elif isinstance(transformer, (BaseEstimator, TransformerMixin)):
            self.other_transformers.append(transformer)
        else:
            raise Exception("Can't append transformer to the Preprocessor")
        return self

    def _to_snake(self, name):
        _name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        _name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", _name).lower()
        return _name

    @property
    def _transformers(self):
        transformers = list(self.text_transformers)
        if self.tokenizer:
            transformers += [self.tokenizer]
        transformers += self.token_transformers
        if self.vocabulary:
            transformers += [self.vocabulary]
        transformers += self.other_transformers

        return (
            (self._to_snake(t.__class__.__name__) + "_at_{}".format(i), t)
            for i, t in enumerate(transformers)
        )

    def _validate_transformers(self):
        names, transformers = zip(*self._transformers)

        # validate names
        self._validate_names(names)

        # validate estimators
        for t in transformers:
            if t is None:
                continue
            if (not (hasattr(t, "fit") or hasattr(t, "fit_transform")) or not
                    hasattr(t, "transform")):
                raise TypeError("All transformer should implement fit and "
                                "transform. '%s' (type %s) doesn't" %
                                (t, type(t)))

    def transform(self, X):
        self._validate_transformers()
        Xt = self.check_array(X, True)
        for name, t in self._transformers:
            Xt = t.transform(Xt)

        return Xt

    def inverse_transform(self, X):
        self._validate_transformers()
        Xt = X
        for name, t in list(self._transformers)[::-1]:
            if hasattr(t, "inverse_transform"):
                Xt = t.inverse_transform(Xt)
        return Xt

    def fit(self, X, y=None):
        self._validate_transformers()
        Xt = X
        copied = False
        for name, t in self._transformers:
            t.fit(Xt)
            if not isinstance(t, Vocabulary):
                original_copy_setting = t.copy
                if not copied:
                    # Do not transform original X
                    t.copy = False
                    copied = True
                else:
                    t.copy = False
                Xt = t.transform(Xt)
                t.copy = original_copy_setting
        return self

    def fit_transform(self, X, y=None):
        self._validate_transformers()
        Xt = X
        for name, t in self._transformers:
            Xt = t.fit_transform(Xt)
        return Xt

    def check_array(self, X, copy_obj=True):
        if not copy_obj:
            return X
        else:
            if isinstance(X, (np.ndarray, np.generic)):
                return np.array(X)
            else:
                return copy.deepcopy(X)

    def save(self, path):
        joblib.dump(self, path)

    @classmethod
    def load(cls, path):
        instance = joblib.load(path)
        return instance
