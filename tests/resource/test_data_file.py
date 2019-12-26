import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from chariot.storage import Storage
from chariot.resource.data_file import DataFile


def resolve(path):
    return os.path.abspath(path)


class TestDataFile(unittest.TestCase):

    def test_convert(self):
        path = os.path.join(os.path.dirname(__file__), "../data")
        storage = Storage(path)
        csv = DataFile(storage.path("raw/sample_dataset.csv"))

        path_changed = csv.convert(data_dir_to="interim")
        correct = os.path.join(path, "./interim/sample_dataset.csv")
        self.assertEqual(resolve(path_changed.path), resolve(correct))

        attr_added = csv.convert(add_attribute="preprocessed")
        correct = storage.path("raw/sample_dataset__preprocessed.csv")
        self.assertEqual(resolve(attr_added.path), resolve(correct))

        attr_converted = attr_added.convert(
                            attribute_to={"preprocessed": "converted"})
        correct = storage.path("raw/sample_dataset__converted.csv")
        self.assertEqual(resolve(attr_converted.path), resolve(correct))

        ext_changed = csv.convert(ext_to=".txt")
        correct = storage.path("raw/sample_dataset.txt")
        self.assertEqual(resolve(ext_changed.path), resolve(correct))

    def test_read_file(self):
        path = os.path.join(os.path.dirname(__file__), "../data")
        storage = Storage(path)
        csv = DataFile(storage.path("raw/sample_dataset.csv"))

        content = csv.to_array()
        fetched = list(csv.fetch(progress=True))
        for c, f in zip(content, fetched):
            self.assertEqual(c, f)


if __name__ == "__main__":
    unittest.main()
