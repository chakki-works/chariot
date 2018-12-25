import re
import copy
import numpy as np
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.base import BaseEstimator, TransformerMixin
from chariot.transformer.vocabulary import Vocabulary
from chariot.transformer.tokenizer import Tokenizer


class Preprocessor(_BaseComposition, BaseEstimator, TransformerMixin):

    def __init__(self, tokenizer=None,
                 text_transformers=(), token_transformers=(),
                 vocabulary=None):
        self.tokenizer = tokenizer
        if isinstance(self.tokenizer, str):
            self.tokenizer = Tokenizer(self.tokenizer)
        self.text_transformers = text_transformers
        self.token_transformers = token_transformers
        self.vocabulary = vocabulary

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

        return (
            (self._to_snake(t.__class__.__name__) + "_", t) for t in
            transformers
        )

    """
    @_transformers.setter
    def _transformers(self, value):
        print("XXXXXXXXXXXXXXXXX")
        for name, t in value:
            if isinstance(t, (TextFilter, TextNormalizer)):
                self.text_transformers.append(t)
            elif isinstance(t, (TokenFilter, TokenNormalizer)):
                self.token_transformers.append(t)
            elif isinstance(t, Tokenizer):
                self.tokenizer = t
            elif isinstance(t, Vocabulary):
                self.vocabulary = t

    def get_params(self, deep=True):
        return self._get_params("_transformers", deep=deep)

    def set_params(self, **kwargs):
        self._set_params("_transformers", **kwargs)
        return self
    """

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
            original_copy_setting = t.copy
            t.copy = False  # Don't copy in each transformer
            Xt = t.transform(Xt)
            t.copy = original_copy_setting
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
        for name, t in self._transformers:
            if isinstance(t, Vocabulary):
                t.fit(Xt)
            else:
                Xt = t.fit_transform(Xt)
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
