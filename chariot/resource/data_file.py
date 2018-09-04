import os
import mmap
from tqdm import tqdm


class DataFile():

    def __init__(self, path, encoding="utf-8"):
        self.path = path
        self.encoding = encoding
        file_name = os.path.basename(path)
        base_name, ext = os.path.splitext(file_name)
        self.base_name = base_name
        self.ext = ext

    def exists(self):
        return os.path.exists(self.path)

    def convert(self, data_dir_to="", add_attribute="",
                attribute_to=(), ext_to=""):
        _dir = os.path.dirname(self.path)
        elements = self._elements()
        ext = self.ext

        if data_dir_to:
            _dir = os.path.join(_dir, "../" + data_dir_to)

        if add_attribute:
            elements.append(add_attribute)
        elif len(attribute_to) > 0:
            # File name format is name + "__".join(attributes)
            # So attribute is elements[1:]
            for a in attribute_to:
                if a in elements[1:]:
                    index = elements[1:].index(a)
                    elements[1 + index] = attribute_to[a]
        if ext_to:
            ext = ext_to

        base_name = "__".join(elements)
        new_path = os.path.join(_dir, base_name + ext)
        return self.__class__(new_path)

    @property
    def name(self):
        return self._elements[0]

    @property
    def attributes(self):
        return self._elements[1:]

    def _elements(self):
        elements = self.base_name.split("__")
        return elements

    def get_line_count(self):
        count = 0
        with open(self.path, "r+") as f:
            buf = mmap.mmap(f.fileno(), 0)
            while buf.readline():
                count += 1
        return count

    def fetch(self, progress=False):
        total_count = 0
        if progress:
            total_count = self.get_line_count()
        with open(self.path, encoding=self.encoding) as f:
            iterator = f
            if progress:
                iterator = tqdm(f, total=total_count)
            for line in iterator:
                yield line.strip()

    def to_array(self):
        lines = []
        with open(self.path, encoding=self.encoding) as f:
            lines = f.readlines()
            lines = [ln.strip() for ln in lines]
        return lines
