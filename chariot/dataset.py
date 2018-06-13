from itertools import chain


class Dataset():

    def __init__(self, data_file, columns):
        self.data_file = data_file
        self.columns = columns

    def __call__(self, *columns):
        for result in self.fetch(target_columns=columns):
            yield result

    def get(self, *columns):
        dataset = {}

        if len(columns) == 0:
            _columns = self.columns
        elif isinstance(columns[0], (list, tuple)):
            _columns = columns[0]
        else:
            _columns = [columns[0]]

        for c in _columns:
            dataset[c] = []

        for result in self.fetch(target_columns=_columns):
            for i, c in enumerate(_columns):
                dataset[c].append(result[i])

        return dataset

    def fetch(self, target_columns=(), progress=False):
        _target = target_columns
        if len(_target) == 0:
            _target = self.columns  # Select all columns

        first = True
        for line in self.data_file.fetch(progress):
            if first and isinstance(line, (list, tuple)):
                # change column name to index
                _target = [i for i, c in enumerate(self.columns)
                           if c in _target]

            if isinstance(line, (dict, list, tuple)):
                result = [line[c] for c in _target]
            else:
                result = line

            first = False
            yield result

    def save_indexed(self, column_processors):
        target_columns = [c for c in self.columns if c in column_processors]
        indexed_columns = []
        dataset = self.get(target_columns)

        for c in column_processors:
            if column_processors[c]:
                dataset[c] = column_processors[c].transform(dataset[c])
                indexed_columns.append(c)

        indexed = self.data_file.convert(data_dir_to="processed",
                                         add_attribute="indexed")

        columns = [dataset[c] for c in target_columns]
        with open(indexed.path, mode="w", encoding=indexed.encoding) as f:
            for elements in zip(*columns):
                strs = []
                for e in elements:
                    if isinstance(e, (list, tuple)):
                        strs.append(" ".join([str(_e) for _e in e]))
                    else:
                        strs.append(str(e))

                line = ""
                if len(strs) > 1:
                    line = indexed.delimiter.join(strs)
                else:
                    line = strs[0]

                f.write(line + "\n")

        indexed_d = IndexedDataset(indexed, target_columns, indexed_columns)
        return indexed_d


class IndexedDataset(Dataset):

    def __init__(self, data_file, columns, indexed_columns):
        super().__init__(data_file, columns)
        self.indexed_columns = indexed_columns

    def fetch(self, target_columns=(), progress=False):
        _target = target_columns
        if len(_target) == 0:
            _target = self.columns

        for result in super().fetch(target_columns, progress):
            _result = []
            for r, t in zip(result, _target):
                if t in self.indexed_columns:
                    # Space separated index string to int array.
                    _result.append([int(i) for i in r.split(" ")])
                else:
                    _result.append(r)
            yield _result
