import csv
from tqdm import tqdm
from chariot.resource.data_file import DataFile


class CsvFile(DataFile):

    def __init__(self, path, encoding="utf-8", delimiter=",",
                 has_header=False):
        super().__init__(path, encoding)
        self.delimiter = delimiter
        self.has_header = has_header
        self.header = []

    def to_dataset(self, fields=()):
        from chariot.dataset import Dataset
        fields = list(fields)
        if len(fields) == 0:
            fields = self.header
        return Dataset(self, fields)

    def fetch(self, progress=False):
        done_header = not self.has_header
        total_count = 0
        if progress:
            total_count = self.get_line_count()

        with open(self.path, encoding=self.encoding) as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            if progress:
                reader = tqdm(reader, total=total_count)

            for row in reader:
                if self.has_header and not done_header:
                    if len(self.header) == 0:
                        self.header = row
                    done_header = True
                else:
                    if len(self.header) > 0:
                        element = {}
                        for name, item in zip(self.header, row):
                            element[name] = item
                        yield element
                    else:
                        yield row

    def write(self, elements_iter):
        with open(self.path, mode="wb", encoding=self.encoding) as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            for elements in elements_iter:
                writer.writerow(elements)
