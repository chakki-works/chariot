import itertools
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from chariot.base_processor import BaseProcessor
from chariot.transformer.formatter.target_generator import TargetGenerator


def _apply(target, process, data, inverse=False):
    input_data = data[target]
    if isinstance(input_data, pd.Series):
        input_data = input_data.values

    if not inverse:
        transformed = process.transform(input_data)
    else:
        if hasattr(process, "inverse_transform"):
            transformed = process.inverse_transform(input_data)
    return (target, transformed)


class Feeder(BaseProcessor):

    def __init__(self, spec):
        super().__init__(spec)
        self._resource = None

    def set_resource(self, data):
        self._resource = data
        return self

    def _get_keys(self, data, ignores):
        if isinstance(data, pd.DataFrame):
            keys = [c for c in data.columns if c not in ignores]
        else:
            keys = [k for k in data if k not in ignores]
        return keys

    def transform(self, data=None, ignores=(), n_jobs=-1, inverse=False):
        _data = data if data else self._resource
        tasks = []
        for k in self.spec:
            if k not in ignores:
                tasks.append((k, self.spec[k]))

        target_transformed = Parallel(n_jobs=n_jobs)(
            delayed(_apply)(i, p, _data, inverse) for i, p in tasks)
        target_transformed = dict(target_transformed)

        applied = {}
        for key in self._get_keys(_data, ignores):
            if key in target_transformed:
                applied[key] = target_transformed[key]
            else:
                applied[key] = _data[key]

        return applied

    def make_generator(self, data, batch_size, epoch=-1, n_jobs=-1,
                       ignores=()):
        _data = data if data else self._resource
        data_length = len(_data)
        steps_per_epoch = data_length // batch_size
        keys = self._get_keys(_data, ignores)

        def generator():
            indices = np.arange(data_length)
            count = 0
            _epoch = 0
            while True:
                if count > 0 and count % steps_per_epoch == 0:
                    _epoch += 1
                    count = 0
                    if epoch > 0 and _epoch >= epoch:
                        break

                selected = np.random.choice(indices, size=batch_size,
                                            replace=False)

                if isinstance(_data, pd.DataFrame):
                    batch = _data[keys].iloc[selected, :]
                else:
                    batch = {}
                    for key in keys:
                        if isinstance(_data[key], pd.Series):
                            batch[key] = _data[key].iloc[selected]
                        else:
                            batch[key] = _data[key][selected]

                count += 1
                yield self.transform(batch, n_jobs=n_jobs)

        return steps_per_epoch, generator

    def iterate(self, data=None, batch_size=32, epoch=-1, n_jobs=-1,
                ignores=()):
        _, generator = self.make_generator(data, batch_size, epoch, n_jobs,
                                           ignores)
        for b in generator():
            yield b

    def inverse_transform(self, batch, n_jobs=-1, ignores=()):
        return self.transform(batch, n_jobs=n_jobs, ignores=ignores, inverse=True)


class LanguageModelFeeder(BaseProcessor):

    def __init__(self, spec):
        super().__init__(spec)
        self._resource = None
        self.key = ""

        if len(spec) != 1:
            raise Exception("Language model should have only 1 column\
                            that contains sentences.")
        else:
            self.key = list(spec.keys())[0]
            if not isinstance(self.spec[self.key], TargetGenerator):
                raise Exception("Language model should take one generator.")

    def set_resource(self, data, batch_size):
        sentences = data[self.key]
        if isinstance(sentences, pd.Series):
            sentences = sentences.values
        elif isinstance(sentences, (list, tuple)):
            np.array(sentences)

        self._resource = np.array(list(itertools.chain.from_iterable(sentences)))
        limit = (len(self._resource) // batch_size) * batch_size
        self._resource = self._resource[:limit].reshape((batch_size, -1)).T

    def make_generator(self, data, batch_size, sequence_length, epoch=-1,
                       sequencial=True):

        self.set_resource(data, batch_size)
        steps_per_epoch = (len(self._resource) - 1) // sequence_length
        _generator = self.spec[self.key]

        def generator():
            count = 0
            _epoch = 0
            while True:
                if count > 0 and count % steps_per_epoch == 0:
                    _epoch += 1
                    count = 0
                    if epoch > 0 and _epoch >= epoch:
                        break
                
                index = count * sequence_length
                data, target = _generator.generate(self._resource, index, sequence_length)
                count += 1
                if sequencial:
                    yield data, target
                else:
                    # Predict one word from sequence
                    yield data.T, target[-1].T

        return steps_per_epoch, generator

    def iterate(self, data, batch_size, sequence_length, epoch=-1, sequencial=True):
        _, generator = self.make_generator(data, batch_size, sequence_length, epoch, sequencial)
        for d, t in generator():
            yield d, t
