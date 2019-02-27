import numpy as np
import pandas as pd
from chariot.transformer.formatter.base import BaseFormatter
from chariot.transformer.generator.base import BaseGenerator
from chariot.preprocessor import Preprocessor
from chariot.base_dataset_preprocessor import BaseDatasetPreprocessor


class ProcessBuilder():

    def __init__(self, dp, name):
        self.dp = dp
        self.key = (name, name)
        if self.key not in self.dp.spec:
            self.dp.spec[self.key] = {}

    @property
    def preprocessor(self):
        if "preprocessor" in self.dp.spec[self.key]:
            return self.dp.spec[self.key]["preprocessor"]
        else:
            return None

    @property
    def formatter(self):
        if "formatter" in self.dp.spec[self.key]:
            return self.dp.spec[self.key]["formatter"]
        else:
            return None

    def fit(self, X, y=None):
        if self.preprocessor is not None:
            self.preprocessor.fit(X, y)
            if self.formatter is not None:
                f = self._convey(self.formatter)
                self.dp.spec[self.key]["formatter"] = f
        return self

    def by(self, processor, reference=None):
        if isinstance(processor, (BaseFormatter, BaseGenerator)):
            p = None if reference is None else reference.preprocessor
            self.dp.spec[self.key]["formatter"] = self._convey(processor, p)
        else:
            if self.preprocessor is None:
                self.dp.spec[self.key]["preprocessor"] = Preprocessor()
            if isinstance(processor, Preprocessor):
                self.dp.spec[self.key]["preprocessor"] = processor
            else:
                self.dp.spec[self.key]["preprocessor"].stack(processor)

        return self

    def as_name(self, to_name):
        new_key = (self.key[0], to_name)
        self.dp.spec[new_key] = self.dp.spec.pop(self.key)
        self.key = new_key
        return ProcessBuilder(self.dp, to_name)

    def _convey(self, formatter, preprocessor=None):
        _p = preprocessor if preprocessor is not None else self.preprocessor
        if _p is not None and hasattr(formatter, "transfer_setting"):
            formatter.transfer_setting(_p)
        return formatter


class DatasetPreprocessor(BaseDatasetPreprocessor):

    def __init__(self, spec=()):
        super().__init__(spec)
        self._processed = None
        self._formatted = False

    @property
    def processed(self):
        return self._processed

    def process(self, name):
        return ProcessBuilder(self, name)

    def get_spec(self, kind):
        spec = {}
        for f in self.spec:
            if kind in self.spec[f]:
                spec[f] = self.spec[f][kind]

        return spec

    def __call__(self, data):
        self._processed = data
        self._formatted = False
        return self

    def preprocess(self, data=None, n_jobs=1, inverse=False,
                   as_dataframe=False):
        spec = self.get_spec("preprocessor")
        _data = data if data is not None else self._processed
        _data = self.transform(spec, _data, n_jobs, inverse, as_dataframe)

        if data is None:
            self._processed = _data
            return self
        else:
            return _data

    def format(self, data=None, n_jobs=1, inverse=False,
               as_dataframe=False):
        spec = self.get_spec("formatter")
        _data = data if data is not None else self._processed
        _data = self.transform(spec, _data, n_jobs, inverse, as_dataframe)

        if data is None:
            self._processed = _data
            self._formatted = True
            return self
        else:
            return _data

    def inverse(self, data, n_jobs=1, as_dataframe=False):
        _data = self.format(data, n_jobs, True, as_dataframe)
        _data = self.preprocess(_data, n_jobs, True, as_dataframe)
        return _data

    def check_length(self, data):
        length = len(data)
        if isinstance(data, dict):
            length = -1
            for k in data:
                if length < 0:
                    length = len(data[k])
                elif length != len(data[k]):
                    raise Exception("Length of each data column is mismatch.")

        return length

    def iterator(self, data=None, batch_size=32, epoch=-1, n_jobs=1,
                 output_epoch_end=False):

        _data = data if data is not None else self._processed
        data_length = self.check_length(_data)
        steps_per_epoch = data_length // batch_size

        def generator():
            indices = np.arange(data_length)
            count = 0
            _epoch = 0
            while True:
                done = False
                if count > 0 and count % steps_per_epoch == 0:
                    done = True
                    _epoch += 1
                    count = 0
                    if epoch > 0 and _epoch >= epoch:
                        break

                selected = np.random.choice(indices, size=batch_size,
                                            replace=False)

                if isinstance(_data, pd.DataFrame):
                    batch = _data.iloc[selected, :]
                else:
                    batch = {}
                    for key in _data:
                        if isinstance(_data[key], pd.Series):
                            batch[key] = _data[key].iloc[selected]
                            batch[key].reset_index(inplace=True, drop=True)
                        else:
                            batch[key] = _data[key][selected]

                count += 1
                if not self._formatted:
                    batch = self.format(batch, n_jobs=n_jobs)
                if output_epoch_end:
                    batch = [batch, done]
                yield batch

        return steps_per_epoch, generator

    def iterate(self, data=None, batch_size=32, epoch=1, n_jobs=1,
                output_epoch_end=False):
        _, iterator = self.iterator(data, batch_size, epoch, n_jobs,
                                    output_epoch_end)
        for batch in iterator():
            yield batch
