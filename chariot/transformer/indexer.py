from collections import Counter
import numbers
from chariot.transformer.base_preprocessor import BasePreprocessor
from chariot.resource.word_vector import WordVector


class Indexer(BasePreprocessor):

    def __init__(self, padding="__PAD__", unknown="__UNK__",
                 begin_of_seq="__BOS__", end_of_seq="__EOS__",
                 min_df=1, size=-1):
        self._vocab = []
        self._padding = padding
        self._unknown = unknown
        self._begin_of_seq = begin_of_seq
        self._end_of_seq = end_of_seq
        self.min_df = min_df
        self.size = size

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
        dataset = []

        # X should be list of tokens or dict
        if isinstance(X, dict):
            for columns in X:
                for tokens in X[columns]:
                    vocab.update(self.token_to_words(tokens))
        else:
            for row in X:
                if len(row) > 0 and isinstance(row[0], (list, tuple)):
                    # row has multiple columns
                    for tokens in row:
                        vocab.update(self.token_to_words(tokens))
                else:
                    vocab.update(self.token_to_words(row))

        reserved = [self._padding, self._unknown,
                    self._begin_of_seq, self._end_of_seq]
        reserved = [r for r in reserved if r]

        selected = reserved
        if self.min_df >= 0:
            md = self.min_df
            min_count = (md if isinstance(md, numbers.Integral)
                         else md * len(dataset))
            for term, count in vocab.most_common():
                if count <= min_count:
                    continue
                else:
                    selected.append(term)
        elif self.size > 0:
            for term, count in vocab.most_common():
                if len(selected) < self.size:
                    selected.append(term)
                else:
                    break

        self._vocab = selected

    def make_embedding(self, word_vector_path,
                       encoding="utf-8", progress=False):
        wv = WordVector(word_vector_path, encoding)
        embedding = wv.load(self.vocab, progress=progress)
        return embedding
