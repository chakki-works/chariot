import itertools
import numpy as np
import pandas as pd
from chariot.dataset_preprocessor import DatasetPreprocessor
from chariot.dataset_preprocessor import ProcessBuilder
from chariot.transformer.formatter.base import BaseFormatter
from chariot.transformer.generator.base import BaseGenerator
from chariot.transformer.generator.target_generator import TargetGenerator


class LanguageModelProcessBuilder(ProcessBuilder):

    def __init__(self, dp, name):
        super().__init__(dp, name)

    def by(self, processor, reference=None):
        if isinstance(processor, BaseGenerator):
            self.dp.generator[self.key] = processor
        else:
            super().by(processor, reference)
        return self


class SentenceToBatchTransformer(BaseFormatter):

    def __init__(self, batch_size):
        self.batch_size = batch_size

    def transform(self, column):
        sentences = column
        if isinstance(column, pd.Series):
            sentences = column.values
        elif isinstance(column, (list, tuple)):
            sentences = np.array(column)

        if len(column.shape) > 1 and column.shape[1] == 1:
            sentences = column.flatten()

        adjusted = np.array(list(itertools.chain.from_iterable(sentences)))
        limit = (len(adjusted) // self.batch_size) * self.batch_size
        # None x batch_size matrix
        adjusted = adjusted[:limit].reshape((self.batch_size, -1)).T
        return adjusted


class LanguageModelPreprocessor(DatasetPreprocessor):

    def __init__(self, spec=()):
        super().__init__(spec)
        self.generator = {}

    def process(self, name):
        return LanguageModelProcessBuilder(self, name)

    def generate(self, column, index=0, sequence_length=12):
        if len(self.generator) == 0:
            raise Exception("Language model should have at least one generator")

        generated_pairs = []
        for k in self.generator:
            from_field, to_field = k
            g = self.generator[k]
            left, right = g.generate(column, index, sequence_length)

            if isinstance(g, TargetGenerator):
                generated_pairs.append((left, right))
            else:
                generated_pairs.append((right, left))

        return generated_pairs

    def iterator(self, data=None, batch_size=32, sequence_length=12,
                 epoch=-1, sequencial=True, output_epoch_end=False,
                 n_jobs=1):

        _data = data if data is not None else self._processed
        for key in self.generator:
            _from, _to = key

        batch_all = SentenceToBatchTransformer(batch_size).transform(_data[_from])
        steps_per_epoch = (len(batch_all) - 1) // sequence_length

        def generator():
            count = 0
            _epoch = 0
            index = 0
            while True:
                done = False
                if count > 0 and count % steps_per_epoch == 0:
                    done = True
                    _epoch += 1
                    count = 0
                    index = 0
                    if epoch > 0 and _epoch >= epoch:
                        break

                pairs = self.generate(batch_all, index, sequence_length)

                if sequencial:
                    batch = pairs
                else:
                    # to (batch x sequence x word_id) shape
                    batch = []
                    for f_, t_ in pairs:
                        batch.append((f_.T, np.expand_dims(t_.T, axis=-1)))

                count += 1
                index = index + sequence_length

                if len(self.generator) > 1:
                    if output_epoch_end:
                        batch = [batch, done]
                else:
                    batch = list(batch[0])
                    if output_epoch_end:
                        batch += [done]

                yield batch

        return steps_per_epoch, generator

    def iterate(self, data=None, batch_size=32, sequence_length=12,
                epoch=1, sequencial=True, output_epoch_end=False,
                n_jobs=1):
        _, generator = self.iterator(data, batch_size, sequence_length,
                                     epoch, sequencial, output_epoch_end,
                                     n_jobs)
        for batch in generator():
            yield batch
