import numpy as np
from itertools import compress


class Batch():

    def __init__(self, dataset, fields, field_adjuster=()):
        self.dataset = dataset
        self.fields = fields
        self.field_adjuster = field_adjuster
        if len(self.field_adjuster) == 0:
            self.field_adjuster = {}

    def full(self):
        batch_data = []
        for f in self.fields:
            adjuster = None
            if f in self.field_adjuster:
                adjuster = self.field_adjuster[f]
            batch_data.append(BatchData(self.dataset[f], adjuster))
        return batch_data

    def iter(self, batch_size, epoch=-1):
        sample_size = 0
        for f in self.fields:
            if sample_size == 0:
                sample_size = len(self.dataset[f])
            elif sample_size != len(self.dataset[f]):
                raise Exception("Data size does not match in dataset.")

        steps_per_epoch = sample_size // batch_size

        def generator():
            candidates = np.zeros(sample_size)
            selected = np.random.randint(batch_size)
            candidates[selected] = 1
            batch = []
            count = 0
            _epoch = 0
            while True:
                if count > 0 and count % steps_per_epoch == 0:
                    _epoch += 1
                    count = 0
                    if epoch > 0 and _epoch > epoch:
                        break
                for f in self.fields:
                    _selected = compress(self.dataset[f], candidates)
                    adjuster = None
                    if f in self.field_adjuster:
                        adjuster = self.field_adjuster[f]
                    batch.append(BatchData(_selected, adjuster))

                count += 1
                yield batch

        return steps_per_epoch, generator


class BatchData():

    def __init__(self, array, adjuster=None):
        self.array = array
        self.adjuster = adjuster

    def __call__(self):
        return np.array(self.array)

    def adjust(self, padding=-1, bos=False, eos=False, to_categorical=False):
        adjusted = [self.adjuster.adjust(x, padding, bos, eos)
                    for x in self.array]

        adjusted = np.array(adjusted)
        if to_categorical:
            return self.adjuster.to_categorical(adjusted)
        else:
            return adjusted
