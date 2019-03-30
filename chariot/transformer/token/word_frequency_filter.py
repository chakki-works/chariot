from chariot.transformer.token.base import TokenFilter
from chariot.util import apply_map
from collections import Counter


class WordFrequencyFilter(TokenFilter):

    def __init__(self, n, min_freq=1, copy=True):
        super().__init__(copy)
        self.n = n
        self.min_freq = min_freq
        self._filter_words = []

    def token_to_words(self, tokens):
        words = [t if isinstance(t, str) else t.surface for t in tokens]
        return [w.strip() for w in words]

    def fit(self, X, y=None):
        words_counter = Counter()
        def update_word_counts(element):
            words = self.token_to_words(element)
            words_counter.update(words)

        apply_map(X, update_word_counts)
        common_words = {word for word, freq in words_counter.most_common(self.n)}
        rare_words = {word for word, freq in words_counter.most_common() if freq <= self.min_freq}
        filter_words = common_words.union(rare_words)
        self._filter_words = filter_words
        return self

    def transform(self, X):
        if len(self._filter_words) == 0:
            raise Exception("Filter words has not made yet. Plase execute fit.")
        return super().transform(X)

    def apply(self, tokens):
        if len(tokens) == 0:
            return tokens
        else:
            return [t for t in tokens if t.surface not in self._filter_words]
