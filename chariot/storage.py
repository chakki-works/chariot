import os
import re
from zipfile import ZipFile
from chariot.resource.data_file import DataFile
from chariot.resource.csv_file import CsvFile


class Storage():

    def __init__(self, root):
        """Provide the access to the data folder.
        The data folder structure should follow the Cookiecutter Data Science
        manner.
        https://drivendata.github.io/cookiecutter-data-science/

        ROOT
         └── data
              ├── external     <- Data from third party sources.
              ├── interim      <- Intermediate data that has been transformed.
              ├── processed    <- The final, canonical data sets for modeling.
              └── raw          <- The original, immutable data dump.

        Args:
            root: path to root directory (parent directory of data, models).
        """
        self.root = root

    def path(self, target=""):
        return os.path.join(self.root, "data/{}".format(target))

    def file(self, target, encoding="utf-8", delimiter=",", has_header=False):
        path = self.path(target)
        _, ext = os.path.splitext(path)
        if ext in [".csv", ".tsv"]:
            return CsvFile(path, encoding, delimiter, has_header)
        else:
            return DataFile(path, encoding)

    def chazutsu(self, path_or_resource, columns=None, target="",
                 separator="\t", pattern=()):

        from chariot.resource.chazutsu_resource import ChazutsuResource
        return ChazutsuResource(path_or_resource, columns,
                                target, separator, pattern)

    def _to_snake(self, name):
        _name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        _name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", _name).lower()
        return _name

    def expand_zip(self, path):
        file_name = os.path.basename(path)
        file_root, ext = os.path.splitext(file_name)
        location = os.path.dirname(path)
        archive_dir = os.path.join(location, file_root)
        if os.path.exists(archive_dir):
            print("The file {} is already expanded".format(
                    os.path.abspath(path)))

        with ZipFile(path) as zip:
            zip.extractall(location)


