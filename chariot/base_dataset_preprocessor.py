import os
import shutil
import json
from ast import literal_eval
from pathlib import Path
import tarfile
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


def _apply(data, target, process, inverse=False):
    if isinstance(data, pd.Series) and data.dtype != "object":
        _data = data.values.reshape((-1, 1))
    elif isinstance(data, np.ndarray) and len(data.shape) < 2:
        _data = data.reshape((-1, 1))
    else:
        _data = data

    if not inverse:
        transformed = process.transform(_data)
    else:
        if hasattr(process, "inverse_transform"):
            transformed = process.inverse_transform(_data)

    return (target, transformed)


class BaseDatasetPreprocessor():

    def __init__(self, spec=()):
        self.spec = spec
        if len(self.spec) == 0:
            self.spec = {}

    def transform(self, spec, data=None, n_jobs=1,
                  inverse=False, as_dataframe=False):
        tasks = []
        keys = []
        if isinstance(data, pd.DataFrame):
            keys = data.columns.tolist()
        else:
            keys = list(data.keys())

        for k in spec:
            from_, to_ = k
            if spec[k] is not None:
                tasks.append((from_, to_, spec[k]))
            if from_ in keys:
                keys.remove(from_)

        transformed = Parallel(n_jobs=n_jobs)(
            delayed(_apply)(data[s], t, p, inverse) for s, t, p in tasks)
        transformed = dict(transformed)

        for k in keys:
            if k not in transformed:
                transformed[k] = data[k]

        if not as_dataframe:
            return transformed
        else:
            applied = pd.DataFrame.from_dict(transformed)
            return applied

    @classmethod
    def _apply_func(cls, d, criteria, func, path=()):
        applied = {}
        for k in d:
            p = list(path) + [k]
            if isinstance(d[k], dict):
                applied[k] = cls._apply_func(d[k], criteria, func, p)
            elif criteria(d[k]):
                applied[k] = func(d[k], p)
            else:
                applied[k] = d[k]

        return applied

    @classmethod
    def _make_file_name(cls, path):
        file_name = ""
        for p in path:
            if isinstance(p, tuple):
                file_name += "-".join(p)
            else:
                file_name += "_" + p
        file_name += ".pkl"
        return file_name

    def save(self, path):
        if os.path.exists(path):
            os.remove(path)
        basename = os.path.basename(path).split(".")[0]
        root = Path(os.path.dirname(path)).joinpath(basename)
        root.mkdir()

        def criteria(item):
            if isinstance(item, (BaseEstimator, TransformerMixin)):
                return True
            else:
                return False

        def func(item, path):
            file_name = self._make_file_name(path)
            joblib.dump(item, root.joinpath(file_name))
            return file_name

        spec = self._apply_func(self.spec, criteria, func)
        with root.joinpath("spec.json").open(mode="w", encoding="utf-8") as f:
            spec = {str(k): v for k, v in spec.items()}
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
            spec = {literal_eval(k): v for k, v in spec.items()}

            def criteria(item):
                if isinstance(item, str) and item.endswith(".pkl"):
                    return True
                else:
                    return False

            def func(item, path):
                file_name = cls._make_file_name(path)
                file_name = basename + "/" + file_name
                p = joblib.load(tar.extractfile(file_name))
                return p

            _spec = cls._apply_func(spec, criteria, func)

        instance = cls(_spec)
        return instance
