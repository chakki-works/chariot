import os
import numpy as np
from sumbase.dataset import Dataset


class ConversionDataset(Dataset):

    def __init__(self, data_file, padding="__PAD__", unknown="__UNK__",
                 begin_of_seq="__BOS__", end_of_seq="__EOS__", vocab_path=""):
        super().__init__(path, "output",
                         delimiter, encoding, has_header)
        self.header = ["input", "output"]
        self.padding = padding
        self.unknown = unknown
        self.begin_of_seq = begin_of_seq
        self.end_of_seq = end_of_seq
        self._vocab_path = vocab_path
        if not self._vocab_path:
            storage = Storage()
            basename = os.path.basename(self.path)
            file_root, ext = os.path.splitext(basename)
            vocab_file = file_root + ".vocab"
            self._vocab_path = storage.data("content/" + vocab_file)
        self._vocab = []
        self._xlim = -1
        self._ylim = -1
        self._padding_cache = -1

    @property
    def unk(self):
        vocab = self.vocab
        return vocab.index(self.unknown)

    @property
    def pad(self):
        if self._padding_cache < 0:
            vocab = self.vocab
            self._padding_cache = vocab.index(self.padding)

        return self._padding_cache

    @property
    def sos(self):
        vocab = self.vocab
        return vocab.index(self.start_of_seq)

    @property
    def eos(self):
        vocab = self.vocab
        return vocab.index(self.end_of_seq)

    def get_Xy_padded(self, x_padding=-1, y_padding=-1):
        Xs, ys = self.get_Xy()
        x_max = max([len(x) for x in Xs])
        y_max = max([len(y) for y in ys])

        self.x_lim = x_padding if x_padding > 0 else x_max
        self.y_lim = y_padding if y_padding > 0 else y_max

        Xs = np.array([self._pad(x, self.x_lim) for x in Xs])
        ys = np.array([self._pad(y, self.y_lim) for y in ys])

        return Xs, ys

    def _pad(self, tokens, limit):
        pad = self.pad
        if len(tokens) < limit:
            pad_size = limit - len(tokens)
            return tokens + [pad] * pad_size
        elif len(tokens) > limit:
            return tokens[:limit]
        return tokens

    def preprocess(self, elements):
        tokens = elements[0].strip().split(" ")
        tokens = [int(t) for t in tokens]
        return tokens

    def to_indices(self, words):
        _words = [w.strip() for w in words]
        tokens = []
        for w in words:
            if w in self.vocab:
                tokens.append(self.vocab.index(w))
            else:
                tokens.append(self.unk)
        return tokens

    def inverse(self, indices, exclude_padding=True):
        vocab = self.vocab
        pad = self.pad
        # Make text exclude padding
        text = [vocab[i] for i in indices]
        if exclude_padding:
            text = [vocab[i] for i in indices if i != pad]
        return "".join(text)
