import os
from pathlib import Path
import shutil
import json
from sklearn.externals import joblib
from chariot.feed import Feed
from chariot.transformer.adjuster import Adjuster


class Dataset():

    def __init__(self, data_file, fields):
        self.data_file = data_file
        self.fields = fields

    def __call__(self, *fields):
        for result in self.fetch(target_fields=fields):
            yield result

    def get(self, *fields):
        dataset = {}

        if len(fields) == 0:
            _fields = self.fields
        elif isinstance(fields[0], (list, tuple)):
            _fields = fields[0]
        else:
            _fields = fields

        for c in _fields:
            dataset[c] = []

        for result in self.fetch(target_fields=_fields):
            for i, c in enumerate(_fields):
                dataset[c].append(result[i])

        return dataset

    def fetch(self, target_fields=(), progress=False):
        _target = target_fields
        if len(_target) == 0:
            _target = self.fields  # Select all fields

        first = True
        for line in self.data_file.fetch(progress):
            if first and isinstance(line, (list, tuple)):
                # change field name to index
                _target = [i for i, c in enumerate(self.fields)
                           if c in _target]

            if isinstance(line, (dict, list, tuple)):
                result = [line[c] for c in _target]
            else:
                result = line

            first = False
            yield result

    def save_transformed(self, transform_name, field_transformers):
        target_fields = [c for c in self.fields if c in field_transformers]
        dataset = self.get(target_fields)

        for f in field_transformers:
            if field_transformers[f]:
                dataset[f] = field_transformers[f].transform(dataset[f])

        transformed = TransformedDataset.create(transform_name, dataset,
                                                field_transformers,
                                                self.data_file)
        return transformed

    def to_feed(self, field_transformers=(),
                apply_transformer=True):
        target_fields = [c for c in self.fields if c in field_transformers]
        dataset = self.get(target_fields)

        field_adjuster = {}
        for f in dataset:
            aj = None
            if f in field_transformers:
                if field_transformers[f] is None:
                    continue
                t = field_transformers[f]
                if apply_transformer:
                    dataset[f] = t.transform(dataset[f])
                if t.indexer is not None:
                    indexer = t.indexer
                    aj = Adjuster(len(indexer.vocab), indexer.pad,
                                  indexer.unk, indexer.bos, indexer.eos)

            if aj is not None:
                field_adjuster[f] = aj

        return Feed(dataset, target_fields, field_adjuster)


class TransformedDataset(Dataset):

    def __init__(self, data_file, original_path, field_transformers):
        fields = list(field_transformers.keys())
        super().__init__(data_file, fields)
        self.original_path = original_path
        self.field_transformers = field_transformers

    @classmethod
    def create(cls, transform_name, data, field_transformers, source_file):
        original_path = source_file.path
        transed = source_file.convert(data_dir_to="processed",
                                      add_attribute=transform_name)
        # Make transformed dir
        root = os.path.join(os.path.dirname(transed.path), transed.base_name)
        root = Path(root)

        if root.exists():
            shutil.rmtree(root)
        root.mkdir()

        # Save transformers
        fields = []
        for f in field_transformers:
            t = field_transformers[f]
            if t is not None:
                joblib.dump(t, root.joinpath("{}.pkl".format(f)))
            fields.append(f)

        # Save Data
        transed = transed.convert(data_dir_to="processed/" + root.name)
        field_data = [data[c] for c in field_transformers]
        with open(transed.path, mode="w", encoding=transed.encoding) as f:
            for elements in zip(*field_data):
                strs = []
                for e in elements:
                    if isinstance(e, (list, tuple)):
                        strs.append(" ".join([str(_e) for _e in e]))
                    else:
                        strs.append(str(e))

                line = ""
                if len(strs) > 1:
                    line = transed.delimiter.join(strs)
                else:
                    line = strs[0]

                f.write(line + "\n")

        # Make metadata
        meta_data = {
            "original": os.path.abspath(original_path),
            "fields": fields
        }

        with root.joinpath("meta.json").open(
                mode="w", encoding="utf-8") as f:
                json.dump(meta_data, f)

        instance = cls(transed, original_path, field_transformers)
        return instance

    @classmethod
    def load(cls, source_file, transform_name):
        transed = source_file.convert(data_dir_to="processed",
                                      add_attribute=transform_name)
        root = os.path.join(os.path.dirname(transed.path),
                            transed.base_name)
        root = Path(root)
        transed_file = transed.convert(data_dir_to="processed/" + root.name)

        with root.joinpath("meta.json").open(encoding="utf-8") as f:
            meta = json.load(f)

        original_path = meta["original"]
        fields = meta["fields"]

        field_transformers = {}
        for f in fields:
            if root.joinpath(f + ".pkl").exists():
                t = joblib.load(root.joinpath(f + ".pkl"))
                field_transformers[f] = t
            else:
                field_transformers[f] = None

        instance = cls(transed_file, original_path, field_transformers)
        return instance

    def fetch(self, target_fields=(), progress=False):
        _target = target_fields
        if len(_target) == 0:
            _target = self.fields

        for result in super().fetch(target_fields, progress):
            _result = []
            for r, t in zip(result, _target):
                if t in self.field_transformers:
                    # Space separated index string to int array.
                    _result.append([int(i) for i in r.split(" ")])
                else:
                    _result.append(r)
            yield _result
