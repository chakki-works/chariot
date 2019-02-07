import itertools
import numpy as np
import pandas as pd
from chariot.dataset_preprocessor import DatasetPreprocessor
from chariot.dataset_preprocessor import ProcessBuilder
from chariot.transformer.formatter.base import BaseFormatter
from chariot.transformer.generator.base import BaseGenerator
from chariot.transformer.generator.target_generator import TargetGenerator


class LanguageModelProcessBuilder(ProcessBuilder):

    def __init__(self, dp, from_name, to_name=""):
        super().__init__(dp, from_name, to_name)

    def by(self, processor):
        if isinstance(processor, BaseFormatter):
            self.dp.spec[self.key]["formatter"] \
                = self.__transfer_setting(processor)
        elif isinstance(processor, BaseGenerator):
            self.dp.generator[self.key] = processor
        else:
            self.dp.spec[self.key]["preprocessor"] = processor
        return self


class LanguageModelPreprocessor(DatasetPreprocessor):

    def __init__(self, spec=()):
        super().__init__(spec)
        self.generator = {}

    def _adjust_data(self, data, batch_size):
        sentences = data[self.key]
        if isinstance(sentences, pd.Series):
            sentences = sentences.values
        elif isinstance(sentences, (list, tuple)):
            sentences = np.array(sentences)

        adjusted = np.array(list(itertools.chain.from_iterable(sentences)))
        limit = (len(adjusted) // batch_size) * batch_size
        # None x batch_size matrix
        adjusted = adjusted[:limit].reshape((batch_size, -1)).T
        return adjusted

    def generate(self, data=None, index=0, sequence_length=12):
        if len(self.generator) == 0:
            raise Exception("Language model should have at least one generator")

        generated_pairs = []
        for k in self.generator:
            from_field, to_field = k
            g = self.generator[k]
            left, right = self.generator(data[from_field], index,
                                         sequence_length)

            if isinstance(g, TargetGenerator):
                generated_pairs.append((left, right))
            else:
                generated_pairs.append((right, left))

        return generated_pairs

    def iterator(self, data, batch_size=32, sequence_length=12,
                 epoch=1, sequencial=True, output_epoch_end=False,
                 n_jobs=1):

        _data = data if data is not None else self._processed
        adjusted = self._adjust_data(_data, batch_size)
        steps_per_epoch = (len(self._resource) - 1) // sequence_length

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

                pairs = self.generate(adjusted, index, sequence_length)

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
        _, generator = self.make_generator(data, batch_size, sequence_length,
                                           epoch, sequencial, output_epoch_end,
                                           n_jobs)
        for batch in generator():
            yield batch
