from collections import Counter
import numbers
from chariot.transformer.base_preprocessor import BasePreprocessor
from chariot.resource.word_vector import WordVector


class Indexer(BasePreprocessor):

    def __init__(self, padding="__PAD__", unknown="__UNK__",
                 begin_of_seq="__BOS__", end_of_seq="__EOS__",
                 max_df=1.0, min_df=1, size=-1, copy=True):
        super().__init__(copy)
        self._vocab = []
        self._padding = padding
        self._unknown = unknown
        self._begin_of_seq = begin_of_seq
        self._end_of_seq = end_of_seq
        self.max_df = max_df
        self.min_df = min_df
        self.size = size

        if max_df < 0 or min_df < 0:
            raise ValueError("Negative value for max_df or min_df")

    def load_vocab(self, list_or_file):
        reserved = [self._padding, self._unknown,
                    self._begin_of_seq, self._end_of_seq]
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

    @property
    def vocab(self):
        return self._vocab

    @property
    def unk(self):
        return self._vocab.index(self._unknown)

    @property
    def pad(self):
        return self._vocab.index(self._padding)

    @property
    def bos(self):
        return self._vocab.index(self._begin_of_seq)

    @property
    def eos(self):
        return self._vocab.index(self._end_of_seq)

    def token_to_words(self, tokens):
        words = [t if isinstance(t, str) else t.surface for t in tokens]
        return [w.strip() for w in words]

    def apply(self, words):
        _words = self.token_to_words(words)
        indices = []
        for w in _words:
            if w in self.vocab:
                indices.append(self.vocab.index(w))
            else:
                indices.append(self.unk)
        return indices

    def inverse(self, indices, exclude_padding=True):
        vocab = self.vocab
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
        return self._transform(X, self.inverse)

    def fit(self, X, y=None):
        vocab = Counter()
        length = 0

        # X should be list of tokens or dict
        if isinstance(X, dict):
            for columns in X:
                for tokens in X[columns]:
                    vocab.update(self.token_to_words(tokens))
                length = max(length, len(X[columns]))
        else:
            for row in X:
                if len(row) > 0 and isinstance(row[0], (list, tuple)):
                    # row has multiple columns
                    for tokens in row:
                        vocab.update(self.token_to_words(tokens))
                else:
                    vocab.update(self.token_to_words(row))
            length = len(X)

        reserved = [self._padding, self._unknown,
                    self._begin_of_seq, self._end_of_seq]
        reserved = [r for r in reserved if r]

        selected = reserved
        if self.size > 0:
            for term, count in vocab.most_common():
                if len(selected) < self.size:
                    selected.append(term)
                else:
                    break
        else:
            min_count = (self.min_df
                         if isinstance(self.min_df, numbers.Integral)
                         else self.min_df * length)
            max_count = (self.max_df
                         if isinstance(self.max_df, numbers.Integral)
                         else self.max_df * length)

            for term, count in vocab.most_common():
                if count <= min_count or count >= max_count:
                    continue
                else:
                    selected.append(term)

        self._vocab = selected

    def make_embedding(self, word_vector_path,
                       encoding="utf-8", progress=False):
        wv = WordVector(word_vector_path, encoding)
        embedding = wv.load(self.vocab, progress=progress)
        return embedding
