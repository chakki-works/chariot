import numpy as np
import pandas as pd
from joblib import Parallel, delayed


def _apply(target, process, data):
    input_data = data[target]
    if isinstance(input_data, pd.Series):
        input_data = input_data.values

    transformed = process.transform(input_data)
    return (target, transformed)


class Feeder():

    def __init__(self, spec):
        self.spec = spec

    def apply(self, data, n_jobs=-1):
        tasks = self.spec.items()

        target_transformed = Parallel(n_jobs=n_jobs)(
            delayed(_apply)(i, p, data) for i, p in tasks)

        applied = {}
        for name, value in target_transformed:
            applied[name] = value

        return applied

    def make_generator(self, data, batch_size, epoch=-1, n_jobs=-1):
        data_length = len(data)
        steps_per_epoch = data_length // batch_size

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

                if isinstance(data, pd.DataFrame):
                    batch = data[list(self.spec.keys())].iloc[selected, :]
                else:
                    batch = {}
                    for key in self.spec:
                        if isinstance(data[key], pd.Series):
                            batch[key] = data[key].iloc[selected]
                        else:
                            batch[key] = data[key][selected]

                count += 1
                yield batch

        return steps_per_epoch, generator

    def iterate(self, data, batch_size, epoch=-1, n_jobs=-1):
        _, generator = self.make_generator(data, batch_size, epoch, n_jobs)
        for b in generator():
            yield b
