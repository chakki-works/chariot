from chariot.resource.csv_file import CsvFile
from chariot.dataset import Dataset
from chazutsu.datasets.framework.resource import Resource


class ChazutsuResource():

    def __init__(self, resource, columns=None, target="",
                 separator="\t", pattern=()):
        if isinstance(resource, Resource):
            self._resource = resource
        else:
            r = Resource(resource, columns, target, separator,
                         pattern)
            self._resource = r

    @property
    def data_file(self):
        return self._get_csv_file("data")

    @property
    def train_file(self):
        return self._get_csv_file("train")

    @property
    def test_file(self):
        return self._get_csv_file("test")

    @property
    def valid_file(self):
        return self._get_csv_file("valid")

    @property
    def sample_file(self):
        return self._get_csv_file("sample")

    @property
    def dataset(self):
        return self._get_dataset("data")

    @property
    def train_dataset(self):
        return self._get_dataset("train")

    @property
    def test_dataset(self):
        return self._get_dataset("test")

    @property
    def valid_dataset(self):
        return self._get_dataset("valid")

    @property
    def sample_dataset(self):
        return self._get_dataset("sample")

    def _get_csv_file(self, kind):
        path = self._resource._get_prop(kind)
        separator = self._resource.separator
        file_instance = CsvFile(path, delimiter=separator, has_header=False)
        if self._resource.columns is None:
            file_instance.header = self._resource.columns
        return file_instance

    def _get_dataset(self, kind):
        data_file = self._get_csv_file(kind)
        fields = data_file.header
        return Dataset(data_file, fields)
