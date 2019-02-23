import os
from pathlib import Path
import shutil
import re
from zipfile import ZipFile
import gzip
import pandas as pd


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
              ├── processed    <- The final, canonical datasets for modeling.
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

    def read(self, target, encoding="utf-8", delimiter=",",
             header="infer", names=None):
        path = self.data_path(target)
        if names is not None:
            df = pd.read_csv(path, delimiter=delimiter, header=None,
                             names=names)
        else:
            df = pd.read_csv(path, delimiter=delimiter, header=header)
        return df

    def chazutsu(self, path, columns=None, target="",
                 separator="\t", pattern=()):

        from chazutsu.datasets.framework.resource import Resource
        r = Resource(path, columns, target, separator, pattern)
        return r

    def chakin(self, lang="", number=-1, name=""):
        import chakin
        if lang:
            chakin.search(lang)
        elif number > -1 or name:
            path = self.data_path("external")
            if not os.path.exists(path):
                os.mkdir(path)

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

            vec_path = chakin.download(index, path)

            base, ext = os.path.splitext(vec_path)
            _dir = os.path.dirname(vec_path)
            if ext == ".vec":
                vec_path = os.rename(vec_path, os.path.join(_dir, _name + ext))
            elif ext in [".zip", ".gz"]:
                _path = self.expand(vec_path, ext)
                os.remove(vec_path)
                vec_path = _path

            return vec_path

        else:
            raise Exception("You have to specify lang to search or "
                            "number/name to download")

    """
    def _to_snake(self, name):
        _name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        _name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", _name).lower()
        return _name
    """

    def expand(self, path, ext):
        location = Path(os.path.dirname(path))
        file_name = os.path.basename(path)
        file_root, ext = os.path.splitext(file_name)

        if ext == ".gz":
            file_path = location.joinpath(file_root)
            with gzip.open(path, "rb") as f_in:
                with open(file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            return file_path
        else:
            dir_path = os.path.join(location, file_root)
            if os.path.exists(dir_path):
                print("The file {} is already expanded".format(
                        os.path.abspath(path)))

            if ext == ".zip":
                with ZipFile(path) as zip:
                    zip.extractall(location)

            return dir_path
