import itertools
import numpy as np
import pandas as pd
from chariot.dataset_preprocessor import DatasetPreprocessor
from chariot.dataset_preprocessor import FieldSpecSetter
from chariot.transformer.generator.target_generator import TargetGenerator
from chariot.transformer.generator.source_generator import SourceGenerator


class LanguageModelFieldSpecSetter(FieldSpecSetter):

    def __init__(self, udp, name):
        super().__init__(udp, name)

    def teacher(self, target_generator):
        self.dp.spec[self.name] = target_generator

    def source(self, source_generator):
        self.dp.spec[self.name] = source_generator


class LanguageModelPreprocessor(DatasetPreprocessor):

    def __init__(self, spec=()):
        super().__init__(spec)

    def verify_spec(self):
        if len(self.spec) != 1:
            raise Exception("Language model should have only 1 column\
                             that contains sentences.")
        else:
            key = list(self.spec.keys())[0]
            generator = self.spec[key]
            if not isinstance(generator, (TargetGenerator, SourceGenerator)):
                raise Exception("Language model should take \
                                 source or target generator.")

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

    def iterator(self, data, batch_size=32, sequence_length=12,
                 epoch=1, sequencial=True, output_epoch_end=False,
                 n_jobs=-1):

        if self.__preprocessed is not None:
            _data = self.__preprocessed
        else:
            _data = self.preprocess(data, n_jobs=n_jobs)

        key = list(self.spec.keys())[0]
        adjusted = self._adjust_data(_data, batch_size)
        steps_per_epoch = (len(self._resource) - 1) // sequence_length

        _generator = self.spec[key]
        is_target_generator = isinstance(_generator, TargetGenerator)

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

                if is_target_generator:
                    data, target = _generator.generate(adjusted, index,
                                                       sequence_length)
                else:
                    target, data = _generator.generate(adjusted, index,
                                                       sequence_length)

                if sequencial:
                    batch = [data, target]
                else:
                    # to (batch x sequence x word_id) shape
                    _target = np.expand_dims(target.T, axis=-1)
                    batch = [data.T, _target]
                if output_epoch_end:
                    batch += [done]

                count += 1
                index = index + sequence_length

                yield batch

        return steps_per_epoch, generator

    def iterate(self, data=None, batch_size=32, sequence_length=12,
                epoch=1, sequencial=True, output_epoch_end=False,
                n_jobs=-1):
        _, generator = self.make_generator(data, batch_size, sequence_length,
                                           epoch, sequencial, output_epoch_end,
                                           n_jobs)
        for batch in generator():
            yield batch
