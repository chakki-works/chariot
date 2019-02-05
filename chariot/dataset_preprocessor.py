import numpy as np
import pandas as pd
from chariot.transformer.formatter.base import BaseFormatter
from chariot.base_dataset_preprocessor import BaseDatasetPreprocessor


class FieldSpecSetter():

    def __init__(self, dp, name):
        self.dp = dp
        self.name = name
        if name not in self.dp.spec:
            self.dp.spec[name] = {}

    def bind(self, processor):
        if isinstance(processor, BaseFormatter):
            self.dp.spec[self.name]["formatter"] \
                = self.__transfer_setting(processor)
        else:
            self.dp.spec[self.name]["preprocessor"] = processor
        return self

    def __transfer_setting(self, formatter):
        if "preprocessor" in self.dp.spec[self.name] and\
           hasattr(formatter, "transfer_setting"):
            p = self.dp.spec[self.name]["preprocessor"]        
            formatter.transfer_setting(p)
        return formatter


class DatasetPreprocessor(BaseDatasetPreprocessor):

    def __init__(self, spec=()):
        super().__init__(spec)
        self.__preprocessed = None

    def field(self, name):
        return FieldSpecSetter(self, name)

    def get_spec(self, kind):
        spec = {}
        for f in self.spec:
            if kind in self.spec[f]:
                spec[f] = self.spec[f][kind]

        return spec

    def preprocess(self, data, n_jobs=-1, inverse=False, as_dataframe=False):
        spec = self.get_spec("preprocessor")
        transformed = self.transform(spec, data, n_jobs, inverse,
                                     as_dataframe)
        self.__preprocessed = transformed
        return transformed

    def preprocessed(self, data):
        self.__preprocessed = data

    def format(self, data=None, n_jobs=-1, inverse=False, as_dataframe=False):
        spec = self.get_spec("formatter")
        _data = data if data is not None else self._processed
        transformed = self.transform(spec, _data, n_jobs, inverse,
                                     as_dataframe)

        return transformed

    def iterator(self, data, batch_size=32, epoch=1, n_jobs=-1,
                 output_epoch_end=False):

        if self.__preprocessed is not None:
            _data = self.__preprocessed
        else:
            _data = self.preprocess(data, n_jobs=n_jobs)

        data_length = len(_data)
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
                    for key in self.spec:
                        if isinstance(_data[key], pd.Series):
                            batch[key] = _data[key].iloc[selected]
                        else:
                            batch[key] = _data[key][selected]

                count += 1
                batch = self.format(batch, n_jobs=n_jobs)
                if output_epoch_end:
                    batch = [batch, done]
                yield batch

        return steps_per_epoch, generator

    def iterate(self, data, batch_size=32, epoch=1, n_jobs=-1,
                output_epoch_end=False):
        _, iterator = self.iterator(data, batch_size, epoch, n_jobs,
                                    output_epoch_end)
        for batch in iterator():
            yield batch
