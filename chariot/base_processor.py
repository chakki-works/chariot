import os
import shutil
import json
from pathlib import Path
import tarfile
from sklearn.externals import joblib


class BaseProcessor():

    def __init__(self, spec):
        self.spec = spec

    def save(self, path):
        basename = os.path.basename(path).split(".")[0]
        root = Path(os.path.dirname(path)).joinpath(basename)
        root.mkdir()

        keymap = {}
        for k in self.spec:
            if isinstance(self.spec[k], dict):
                keymap[k] = {}
                for j in self.spec[k]:
                    keymap[k][j] = "_".join([k, j]) + ".pkl"
                    joblib.dump(self.spec[k][j], root.joinpath(keymap[k][j]))
            else:
                keymap[k] = k + ".pkl"
                joblib.dump(self.spec[k], root.joinpath(keymap[k]))

        with root.joinpath("spec.json").open(mode="w", encoding="utf-8") as f:
            json.dump(keymap, f)

        with tarfile.open(path, "w:gz") as tar:
            tar.add(root, arcname=basename)

        shutil.rmtree(root)

    @classmethod
    def load(cls, path):
        basename = os.path.basename(path).split(".")[0]
        with tarfile.open(path, "r:gz") as tar:
            spec_f = tar.extractfile(basename + "/spec.json")
            spec = json.load(spec_f)
            for k in spec:
                if isinstance(spec[k], dict):
                    for j in spec[k]:
                        f_name = "_".join([k, j]) + ".pkl"
                        f_name = basename + "/" + f_name
                        p = joblib.load(tar.extractfile(f_name))
                        spec[k][j] = p
                else:
                    f_name = k + ".pkl"
                    f_name = basename + "/" + f_name
                    spec[k] = joblib.load(tar.extractfile(f_name))

        instance = cls(spec)
        return instance

    def inverse_transform(self, X):
        self._validate_transformers()
        Xt = X
        for name, t in list(self._transformers)[::-1]:
            if hasattr(t, "inverse_transform"):
                Xt = t.inverse_transform(Xt)
        return Xt
