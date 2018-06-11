import os
import mmap
from tqdm import tqdm
from sumbase.storage import Storage


class DataFile():

    def __init__(self, path):
        self.path = path
        file_name = os.path.basename(path)
        base_name, ext = os.path.splitext(file_name)
        self.base_name = base_name

    @classmethod
    def create(cls, root, target, name, attribute, kind, ext):
        file_name = "__".join([name, attribute, kind]) + ext
        storage = Storage(root)
        _dir = storage.data(target)
        path = os.path.join(_dir, file_name)
        return cls(path)

    def exists(self):
        return os.path.exists(self.path)

    def convert(self, data_dir_to="", attribute_to="", ext_to=""):
        _dir = os.path.dirname(self.path)
        file_name = os.path.basename(self.path)
        base_name, ext = os.path.splitext(file_name)
        if attribute_to:
            base_array = [self.domain, attribute_to, self.kind]
            base_name = "__".join(base_array)
        if ext_to:
            ext = ext_to

        if data_dir_to:
            storage = Storage()
            _dir = storage.data(data_dir_to)

        new_path = os.path.join(_dir, base_name + ext)
        return self.__class__(new_path)

    @property
    def domain(self):
        return self.__get_attr(0)

    @property
    def attribute(self):
        return self.__get_attr(1)

    @property
    def kind(self):
        return self.__get_attr(2)

    def __get_attr(self, index):
        tokens = self.base_name.split("__")
        if len(tokens) < index:
            return None
        else:
            return tokens[index]

    def get_line_count(self):
        count = 0
        with open(self.path, "r+") as f:
            buf = mmap.mmap(f.fileno(), 0)
            while buf.readline():
                count += 1
        return count

    def fetch(self, encoding="utf-8"):
        total_count = self.get_line_count()
        with open(self.path) as f:
            for line in tqdm(f, total=total_count):
                yield line
