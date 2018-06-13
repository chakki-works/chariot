import os
import numbers
from collections import Counter
import numpy as np


class Corpus():

    def __init__(self, data_file, vocab_file=None,
                 label_column=0, column_setting=(),
                 padding="__PAD__", unknown="__UNK__",
                 begin_of_seq="__BOS__", end_of_seq="__EOS__"):
        self.data_file = data_file
        self.vocab_file = vocab_file
        if self.vocab_file is None:
            self.vocab_file = self.data_file.convert(ext_to=".vocab")
        self.label_column = label_column
        self.column_setting = column_setting
        self._padding = padding
        self._unknown = unknown
        self._begin_of_seq = begin_of_seq
        self._end_of_seq = end_of_seq
        self._vocab = []
        self._cache = {}

    @classmethod
    def build(cls, data_file, parser,  label_column=0,
              column_setting=(), min_df=1, progress=True,
              padding="__PAD__", unknown="__UNK__",
              begin_of_seq="__BOS__", end_of_seq="__EOS__"):

        vocab = Counter()
        dataset = []

        def parse(text):
            return parser.parse(text, return_surface=True)

        _column_setting = column_setting
        for line in data_file.fetch(progress):
            if isinstance(line, str):
                line = [line]

            if len(_column_setting) == 0:
                if isinstance(line, (tuple, list)):
                    _column_setting = {i: True for i in range(len(line))}
                elif isinstance(line, dict):
                    _column_setting = {k: True for k in line}

            features = {k: parse(line[k]) if _column_setting[k] else line[k]
                        for k in _column_setting}

            for k in features:
                if _column_setting[k]:
                    for t in features[k]:
                        vocab[t] += 1
            dataset.append(features)

        min_count = (min_df if isinstance(min_df, numbers.Integral)
                     else min_df * len(dataset))

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
            for features in dataset:
                elements = []
                for k in _column_setting:
                    if _column_setting[k]:
                        tokens = features[k]
                        string = " ".join([str(get_index(t)) for t in tokens])
                        elements.append(string)
                    else:
                        elements.append(features[k])

                line = ""
                if len(elements) > 1:
                    line = data_file.delimiter.join(elements)
                else:
                    line = elements[0]
                f.write(line + "\n")

        return cls(indexed_file, label_column=label_column,
                   column_setting=_column_setting,
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

    def to_dataset(self, progress=False, word_sequence=False,
                   label_format_func=None, feature_format_func=None):
        labels = []
        features = {}

        for f, lb in self.fetch(progress=progress,
                                word_sequence=word_sequence):

            labels.append(lb)
            for k in f:
                if k not in features:
                    features[k] = []
                features[k].append(f[k])

        if label_format_func:
            labels = label_format_func(labels)
        else:
            labels = np.array(labels)

        if feature_format_func:
            keys = list(features.keys())
            _func_dict = {}
            if isinstance(feature_format_func, (list, tuple)):
                for k, f in zip(keys, feature_format_func):
                    _func_dict[k] = f
            elif isinstance(feature_format_func, dict):
                _func_dict = feature_format_func
            else:
                for k in keys:
                    _func_dict[k] = feature_format_func

            for k in keys:
                if self.column_setting[k]:
                    features[k] = _func_dict[k](features[k])

        if len(features) == 1:
            features = list(features.values())[0]
        else:
            features = [v for k, v in sorted(features.items())]

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
