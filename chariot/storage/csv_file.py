import csv
from tqdm import tqdm
from sumbase.storage.data_file import DataFile


class CsvFile(DataFile):

    def __init__(self, path, delimiter=","):
        super().__init__(path)
        self.delimiter = delimiter
        self.header = []

    @classmethod
    def create(cls, target, name, attribute, kind):
        return super().create(target, name, attribute, kind, ".csv")

    def fetch(self, encoding="utf-8", has_header=False):
        total_count = self.get_line_count()
        done_header = not has_header
        with open(self.path, encoding=encoding) as f:
            reader = csv.reader(f, delimiter=self.delimiter)
            for row in tqdm(reader, total=total_count):
                if has_header and not done_header:
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
        with open(self.path, mode="wb", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter=self.delimiter)
            for elements in elements_iter:
                writer.writerow(elements)
