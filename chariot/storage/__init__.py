import os
import re
from datetime import datetime
from zipfile import ZipFile


class Storage():

    def __init__(self, root):
        """Provide the access to the data folder.
        The data folder structure should follow the Cookiecutter Data Science
        manner.
        https://drivendata.github.io/cookiecutter-data-science/

        ├── data
        │   ├── external     <- Data from third party sources.
        │   ├── interim      <- Intermediate data that has been transformed.
        │   ├── processed    <- The final, canonical data sets for modeling.
        │   ├── raw          <- The original, immutable data dump.
        │   └── log          <- [ADD] Training results are stored.
        ├── models           <- Trained and serialized models

        Args:
            root: path to root directory (parent directory of data, models).
        """
        self.root = root

    def data(self, target=""):
        return os.path.join(self.root, "data/{}".format(target))

    def model(self, target_or_instance, extention=".h5"):
        _target = target_or_instance
        if not isinstance(target_or_instance, str):
            name = self._to_snake(target_or_instance.__class__.__name__)
            _target = name + extention
        return os.path.join(self.root, "models/{}".format(_target))

    def make_logdir(self, instance):
        name = self._to_snake(instance.__class__.__name__)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        _dir = os.path.join(self.root, "data/log")
        for n in (name, timestamp):
            _dir = os.path.join(_dir, n)
            if not os.path.exists(_dir):
                os.mkdir(_dir)

        return _dir, name

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
