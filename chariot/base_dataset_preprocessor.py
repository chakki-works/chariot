import os
import shutil
import json
from pathlib import Path
import tarfile
import pandas as pd
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


def _apply(data, target, process, inverse=False):
    input_data = data[target]
    if isinstance(input_data, pd.Series):
        input_data = input_data.values.reshape((-1, 1))

    if not inverse:
        transformed = process.transform(input_data)
    else:
        if hasattr(process, "inverse_transform"):
            transformed = process.inverse_transform(input_data)

    return (target, transformed)


class BaseDatasetPreprocessor():

    def __init__(self, spec=()):
        self.spec = spec
        if len(self.spec) == 0:
            self.spec = {}

    def transform(self, spec, data=None, n_jobs=-1,
                  inverse=False, as_dataframe=False):
        tasks = []

        for k in spec:
            if spec[k] is not None:
                tasks.append((k, spec[k]))

        transformed = Parallel(n_jobs=n_jobs)(
            delayed(_apply)(data, t, p, inverse) for t, p in tasks)
        transformed = dict(transformed)

        for key in self.spec:
            if key not in transformed and key not in spec:
                transformed[key] = data[key]

        if not as_dataframe:
            return transformed
        else:
            applied = pd.DataFrame.from_dict(transformed)
            return applied

    @classmethod
    def _apply_func(cls, d, criteria, func, path=()):
        applied = {}
        _path = list(path)
        for k in d:
            _path.append(k)
            if isinstance(d[k], dict):
                applied[k] = cls._apply_func(d[k], criteria, func, _path)
            elif criteria(d[k]):
                applied[k] = func(d[k], _path)
            else:
                applied[k] = d[k]

        return applied

    def save(self, path):
        basename = os.path.basename(path).split(".")[0]
        root = Path(os.path.dirname(path)).joinpath(basename)
        root.mkdir()

        def criteria(item):
            if isinstance(item, (BaseEstimator, TransformerMixin)):
                return True
            else:
                return False

        def func(item, path):
            file_name = "_".join(path) + ".pkl"
            joblib.dump(item, root.joinpath(file_name))
            return file_name

        spec = self._apply_func(self.spec, criteria, func)
        with root.joinpath("spec.json").open(mode="w", encoding="utf-8") as f:
            json.dump(spec, f)

        with tarfile.open(path, "w:gz") as tar:
            tar.add(root, arcname=basename)

        shutil.rmtree(root)

    @classmethod
    def load(cls, path):
        basename = os.path.basename(path).split(".")[0]

        with tarfile.open(path, "r:gz") as tar:
            spec_f = tar.extractfile(basename + "/spec.json")
            spec = json.load(spec_f)

            def criteria(item):
                if isinstance(item, str) and item.endswith(".pkl"):
                    return True
                else:
                    return False

            def func(item, path):
                file_name = "_".join(path) + ".pkl"
                file_name = basename + "/" + file_name
                p = joblib.load(tar.extractfile(file_name))
                return p

            _spec = cls._apply_func(spec, criteria, func)

        instance = cls(_spec)
        return instance
