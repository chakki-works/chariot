import os
import unittest
from sumbase.storage import Storage


def resolve(path):
    return os.path.abspath(path)


class TestStorage(unittest.TestCase):

    def test_path(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        storage = Storage(root)
        correct_path = os.path.join(root, "data/raw")
        self.assertEqual(resolve(storage.data("raw")), resolve(correct_path))


if __name__ == "__main__":
    unittest.main()
