from sumbase.storage.csv_file import CsvFile


class Dataset(CsvFile):

    def __init__(self, path, label_column,
                 delimiter=",", encoding="utf-8", has_header=False):
        super().__init__(path)
        self.label_column = label_column
        self.delimiter = delimiter
        self.encoding = encoding
        self.has_header = has_header
        self._dataset = ()

    def get_Xy(self):
        if len(self._dataset) > 0:
            return self._dataset[0], self._dataset[1]

        Xs = []
        ys = []
        for element in self.fetch(self.delimiter,
                                  self.encoding, self.has_header):
            X = [element[c] for c in self.header if c != self.label_column]
            y = [element[self.label_column]]
            Xs.append(self.preprocess(X))
            ys.append(self.preprocess(y))

        self._dataset = (Xs, ys)
        return self._dataset[0], self._dataset[1]

    def preprocess(self, line):
        return line
