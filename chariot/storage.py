import os
from pathlib import Path
import shutil
import re
from zipfile import ZipFile
import gzip
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

    @classmethod
    def setup_data_dir(cls, path):
        if not os.path.exists(path):
            raise Exception("{} does not exist".format(path))

        root = Path(path)
        data_root = root.joinpath("data")
        data_root.mkdir(exist_ok=True)
        for _dir in ["raw", "external", "interim", "processed"]:
            data_root.joinpath(_dir).mkdir(exist_ok=True)

        storage = cls(path)
        return storage

    def data_path(self, target=""):
        return os.path.join(self.root, "data/{}".format(target))

    def file(self, target, encoding="utf-8", delimiter=",", has_header=False):
        path = self.data_path(target)
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

    def chakin(self, lang="", number=-1, name=""):
        import chakin
        if lang:
            chakin.search(lang)
        elif number > -1 or name:
            path = self.data_path("external")
            table = chakin.downloader.load_datasets()

            index = number
            if number < 0:
                index = table.index[table["Name"] == name].tolist()
                index = index[0]

            _name = table.iloc[index]["Name"].lower()

            for ext in [".txt", ".vec"]:
                check_path = os.path.join(path, _name) + ext
                if os.path.exists(check_path):
                    return check_path

            path = chakin.download(index, path)

            base, ext = os.path.splitext(path)
            _dir = os.path.dirname(path)
            if ext == ".vec":
                path = os.rename(path, os.path.join(_dir, _name + ext))
            elif ext in [".zip", ".gz"]:
                _path = self.expand(path, ext)
                os.remove(path)
                path = _path

            return path

        else:
            raise Exception("You have to specify lang to search or "
                            "number/name to download")

    def _to_snake(self, name):
        _name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        _name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", _name).lower()
        return _name

    def expand(self, path, ext):
        file_name = os.path.basename(path)
        file_root, ext = os.path.splitext(file_name)

        if ext == ".gz":
            with gzip.open(path, "rb") as f_in:
                with open(file_root, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return file_root
        else:
            location = os.path.dirname(path)
            archive_dir = os.path.join(location, file_root)
            if os.path.exists(archive_dir):
                print("The file {} is already expanded".format(
                        os.path.abspath(path)))

            if ext == ".zip":
                with ZipFile(path) as zip:
                    zip.extractall(location)

            return location
