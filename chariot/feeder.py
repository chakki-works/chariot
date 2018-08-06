import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from chariot.base_processor import BaseProcessor


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

    def _get_keys(self, data, ignore):
        if isinstance(data, pd.DataFrame):
            keys = [c for c in data.columns if c not in ignore]
        else:
            keys = [k for k in data if k not in ignore]
        return keys

    def apply(self, data=None, ignore=(), n_jobs=-1, inverse=False):
        _data = data if data else self._resource
        tasks = []
        for k in self.spec:
            if k not in ignore:
                tasks.append((k, self.spec[k]))

        target_transformed = Parallel(n_jobs=n_jobs)(
            delayed(_apply)(i, p, _data, inverse) for i, p in tasks)
        target_transformed = dict(target_transformed)

        applied = {}
        for key in self._get_keys(_data, ignore):
            if key in target_transformed:
                applied[key] = target_transformed[key]
            else:
                applied[key] = _data[key]

        return applied

    def make_generator(self, data, batch_size, epoch=-1, n_jobs=-1,
                       ignore=()):
        _data = data if data else self._resource
        data_length = len(_data)
        steps_per_epoch = data_length // batch_size
        keys = self._get_keys(_data, ignore)

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
                yield self.apply(batch, n_jobs=n_jobs)

        return steps_per_epoch, generator

    def iterate(self, data=None, batch_size=32, epoch=-1, n_jobs=-1,
                ignore=()):
        _, generator = self.make_generator(data, batch_size, epoch, n_jobs,
                                           ignore)
        for b in generator():
            yield b

    def inverse_transform(self, batch, n_jobs=-1, ignore=()):
        return self.apply(batch, n_jobs=n_jobs, ignore=ignore, inverse=True)
