class Dataset():

    def __init__(self, data_file, padding="__PAD__", unknown="__UNK__",
                 begin_of_seq="__BOS__", end_of_seq="__EOS__"):
        self._data_file = data_file
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
