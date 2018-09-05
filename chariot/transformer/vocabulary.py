from collections import Counter
import numbers
from chariot.util import apply_map
from chariot.transformer.base_preprocessor import BasePreprocessor
from chariot.resource.word_vector import WordVector


class Vocabulary(BasePreprocessor):

    def __init__(self, padding="@@PADDING@@", unknown="@@UNKNOWN@@",
                 begin_of_sequence="@@BEGIN_OF_SEQUENCE@@",
                 end_of_sequence="@@END_OF_SEQUENCE@@",
                 max_df=1.0, min_df=1, limit=-1, copy=True):
        super().__init__(copy)
        self._vocab = []
        self._padding = padding
        self._unknown = unknown
        self._begin_of_sequence = begin_of_sequence
        self._end_of_sequence = end_of_sequence
        self.max_df = max_df
        self.min_df = min_df
        self.limit = limit

        if max_df < 0 or min_df < 0:
            raise ValueError("Negative value for max_df or min_df")

    @classmethod
    def from_file(cls, path, padding="@@PADDING@@", unknown="@@UNKNOWN@@",
                  begin_of_sequence="", end_of_sequence="",
                  max_df=1.0, min_df=1, limit=-1, copy=True):

        instance = cls(padding, unknown, begin_of_sequence, end_of_sequence,
                       max_df, min_df, limit, copy)

        with open(path, encoding="utf-8") as f:
            words = f.readlines()
            words = [w.strip() for w in words]
        instance._vocab = words
        return instance

    def set(self, list_or_file):
        reserved = [self._padding, self._unknown,
                    self._begin_of_sequence, self._end_of_sequence]
        reserved = [r for r in reserved if r]

        if isinstance(list_or_file, (list, tuple)):
            vocab = reserved + list(list_or_file)
        else:
            with open(list_or_file, encoding="utf-8") as f:
                words = f.readlines()
                words = [w.strip() for w in words]
            reserved = [r for r in reserved if r not in words]
            vocab = reserved + words

        self._vocab = vocab

    def get(self):
        return self._vocab

    @property
    def count(self):
        return len(self._vocab)

    @property
    def unk(self):
        return self._vocab.index(self._unknown)

    @property
    def pad(self):
        return self._vocab.index(self._padding)

    @property
    def bos(self):
        return self._vocab.index(self._begin_of_sequence)

    @property
    def eos(self):
        return self._vocab.index(self._end_of_sequence)

    def token_to_words(self, tokens):
        words = [t if isinstance(t, str) else t.surface for t in tokens]
        return [w.strip() for w in words]

    def apply(self, words):
        _words = self.token_to_words(words)
        indices = []
        for w in _words:
            if w in self._vocab:
                indices.append(self._vocab.index(w))
            else:
                indices.append(self.unk)
        return indices

    def inverse(self, indices, exclude_padding=True):
        vocab = self._vocab
        pad = self.pad
        # Make text exclude padding
        words = [vocab[i] for i in indices]
        if exclude_padding:
            words = [vocab[i] for i in indices if i != pad]
        return words

    def transform(self, X):
        if len(self._vocab) == 0:
            raise Exception("Vocabulary has not made yet. Plase execute fit.")
        return super().transform(X)

    def inverse_transform(self, X):
        if len(self._vocab) == 0:
            raise Exception("Vocabulary has not made yet. Plase execute fit.")
        return apply_map(X, self.inverse, inplace=self.copy)

    def fit(self, X, y=None):
        vocab = Counter()
        length = len(X)
        if isinstance(X, dict):
            length = len(list(X.values)[0])

        def update_vocab(element):
            vocab.update(self.token_to_words(element))

        apply_map(X, update_vocab)

        reserved = [self._padding, self._unknown,
                    self._begin_of_sequence, self._end_of_sequence]
        reserved = [r for r in reserved if r]  # filter no setting token

        selected = reserved
        if self.limit > 0:
            for term, limit in vocab.most_common():
                if len(selected) < self.limit:
                    selected.append(term)
                else:
                    break
        else:
            min_limit = (self.min_df
                         if isinstance(self.min_df, numbers.Integral)
                         else self.min_df * length)
            max_limit = (self.max_df
                         if isinstance(self.max_df, numbers.Integral)
                         else self.max_df * length)

            for term, limit in vocab.most_common():
                if limit < min_limit or limit > max_limit:
                    continue
                else:
                    selected.append(term)

        self._vocab = selected

    def make_embedding(self, word_vector_path,
                       encoding="utf-8", progress=False):
        wv = WordVector(word_vector_path, encoding)
        embedding = wv.load(self._vocab, progress=progress)
        return embedding
