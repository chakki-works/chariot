import os
import numbers
from collections import Counter
import numpy as np


class Corpus():

    def __init__(self, data_file, vocab_file=None,
                 label_column=0, feature_columns=(),
                 padding="__PAD__", unknown="__UNK__",
                 begin_of_seq="__BOS__", end_of_seq="__EOS__"):
        self.data_file = data_file
        self.vocab_file = vocab_file
        if self.vocab_file is None:
            self.vocab_file = self.data_file.convert(ext_to=".vocab")
        self.label_column = label_column
        self.feature_columns = feature_columns
        self._padding = padding
        self._unknown = unknown
        self._begin_of_seq = begin_of_seq
        self._end_of_seq = end_of_seq
        self._vocab = []
        self._cache = {}

    @classmethod
    def build(cls, data_file, parser, target_columns=(),
              label_column=0, min_df=1, progress=True,
              padding="__PAD__", unknown="__UNK__",
              begin_of_seq="__BOS__", end_of_seq="__EOS__"):

        vocab = Counter()
        tokenized = []

        def parse(text):
            return parser.parse(text, return_surface=True)

        for line in data_file.fetch(progress):
            if isinstance(line, str):
                features = [parse(line)]
            elif isinstance(line, (tuple, list)):
                if len(target_columns) > 0:
                    keys = target_columns
                else:
                    keys = list(range(len(line)))

                features = [parse(ln) for i, ln in enumerate(line)
                            if i in keys]
            elif isinstance(line, dict):
                if len(target_columns) > 0:
                    keys = target_columns
                else:
                    keys = list(line.keys())

                features = [parse(ln) for i, ln in enumerate(line)
                            if i in keys]

            for tokens in features:
                for t in tokens:
                    vocab[t] += 1
            tokenized.append(features)

        min_count = (min_df if isinstance(min_df, numbers.Integral)
                     else min_df * len(tokenized))

        selected = []
        for term, count in vocab.most_common():
            if count <= min_count:
                continue
            else:
                selected.append(term)

        selected = [padding, unknown, begin_of_seq, end_of_seq] + selected

        # Write vocabulary file
        encoding = data_file.encoding
        vocab_file = data_file.convert(data_dir_to="processed",
                                       add_attribute="indexed",
                                       ext_to=".vocab")
        _dir = os.path.dirname(vocab_file.path)
        if not os.path.exists(_dir):
            os.mkdir(_dir)
        with open(vocab_file.path, mode="w", encoding=encoding) as f:
            for v in selected:
                f.write((v + "\n"))

        # Write indexed file
        indexed_file = vocab_file.convert(ext_to=data_file.ext)

        def get_index(t):
            if t in selected:
                return selected.index(t)
            else:
                return selected.index(unknown)

        with open(indexed_file.path, mode="w", encoding=encoding) as f:
            for features in tokenized:
                elements = []
                for tokens in features:
                    string = " ".join([str(get_index(t)) for t in tokens])
                    elements.append(string)

                line = ""
                if len(elements) > 1:
                    line = data_file.delimiter.join(elements)
                else:
                    line = elements[0]
                f.write(line + "\n")

        return cls(indexed_file, label_column=label_column,
                   padding=padding, unknown=unknown,
                   begin_of_seq=begin_of_seq, end_of_seq=end_of_seq)

    @property
    def vocab(self):
        if len(self._vocab) == 0:
            self._load_vocab()
        return self._vocab

    @property
    def unk(self):
        if len(self._cache) == 0:
            self._load_vocab()
        return self._cache[self._unknown]

    @property
    def pad(self):
        if len(self._cache) == 0:
            self._load_vocab()
        return self._cache[self._padding]

    @property
    def bos(self):
        if len(self._cache) == 0:
            self._load_vocab()
        return self._cache[self._begin_of_seq]

    @property
    def eos(self):
        if len(self._cache) == 0:
            self._load_vocab()
        return self._cache[self._end_of_seq]

    def _load_vocab(self):
        self._vocab = self.vocab_file.to_array()
        self._cache = {}
        for token in [self._padding, self._unknown,
                      self._begin_of_seq, self._end_of_seq]:
            self._cache[token] = self._vocab.index(token)

    def words_to_indices(self, words):
        _words = [w.strip() for w in words]
        tokens = []
        for w in _words:
            if w in self.vocab:
                tokens.append(self.vocab.index(w))
            else:
                tokens.append(self.unk)
        return tokens

    def text_to_indices(self, text, word_sequence=False):
        tokens = text.strip().split(" ")
        if word_sequence:
            indices = self.words_to_indices(tokens)
        else:
            indices = [int(i) for i in tokens]
        return indices

    def format(self, indices, padding=-1, bos=False, eos=False):
        if bos:
            indices = [self.bos] + indices
        if eos:
            indices = indices + [self.eos]

        if padding > 0:
            pad = self.pad
            if len(indices) < padding:
                pad_size = padding - len(indices)
                indices = indices + [pad] * pad_size
            elif len(indices) > padding:
                indices = indices[:padding]

        return indices

    def inverse(self, indices, exclude_padding=True, join_str=" "):
        vocab = self.vocab
        pad = self.pad
        # Make text exclude padding
        text = [vocab[i] for i in indices]
        if exclude_padding:
            text = [vocab[i] for i in indices if i != pad]
        return join_str.join(text)

    def fetch(self, progress=False, word_sequence=False):
        wq = word_sequence
        for line in self.data_file.fetch(progress):
            label = []
            features = []
            if isinstance(line, str):
                features = self.text_to_indices(line, wq)
            elif isinstance(line, (list, tuple)):
                if not isinstance(self.label_column, int):
                    raise Exception("You have to specify columns by integer.")
                label = self.text_to_indices(line[self.label_column], wq)
                if len(self.feature_columns) == 0:
                    features = [e for i, e in enumerate(line)
                                if i != self.label_column]
                else:
                    features = [e for i, e in line
                                if i in self.feature_columns]

                if len(features) == 1:
                    features = self.text_to_indices(features[0], wq)
                else:
                    features = [self.text_to_indices(f, wq) for f in features]

            elif isinstance(line, dict):
                if not isinstance(self.label_column, str):
                    raise Exception("You have to specify columns by str.")

                label = self.text_to_indices(line[self.label_column], wq)
                if len(self.feature_columns) == 0:
                    features = [line[k] for k in line
                                if k != self.label_column]
                else:
                    features = [line[k] for k in line
                                if k in self.feature_columns]

                if len(features) == 1:
                    features = self.text_to_indices(features[0], wq)
                else:
                    features = [self.text_to_indices(f, wq) for f in features]

            yield features, label

    def to_dataset(self, progress=False, word_sequence=False,
                   label_format_func=None, feature_format_func=None):
        labels = []
        features = []
        multiple_feature = False

        for f, lb in self.fetch(progress=progress,
                                word_sequence=word_sequence):
            if all(isinstance(_f, list) for _f in f):
                if len(features) == 0:
                    multiple_feature = True
                    for i in range(len(f)):
                        features.append([])
                for i, _f in enumerate(f):
                    features[i].append(_f)
            else:
                features.append(f)
            labels.append(lb)

        if label_format_func:
            labels = label_format_func(labels)
        else:
            labels = np.array(labels)

        if feature_format_func:
            if multiple_feature:
                if not isinstance(feature_format_func, (list, tuple)):
                    raise Exception("Feature format func is insufficient.")
                elif len(features) != len(feature_format_func):
                    raise Exception("Feature format func is insufficient.")

                for i in range(len(features)):
                    features[i] = feature_format_func[i](features[i])

            else:
                features = feature_format_func(features)

        return features, labels

    def format_func(self, padding=-1, bos=False, eos=False,
                    to_categorical=False):

        def _format(indices_list):
            formatted = [self.format(ids, padding, bos, eos)
                         for ids in indices_list]
            formatted = np.array(formatted)
            if to_categorical:
                formatted = self.to_categorical(formatted)

            return formatted

        return _format

    def to_categorical(self, array, size=-1):
        size = size
        if size < 0:
            size = len(self.vocab)
        y = np.array(array, dtype="int")
        input_shape = y.shape
        y = y.ravel()
        n = y.shape[0]
        categorical = np.zeros((n, size), dtype=np.float32)
        categorical[np.arange(n), y] = 1
        output_shape = input_shape + (size,)
        categorical = np.reshape(categorical, output_shape)
        return categorical
