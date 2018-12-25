import os
import shutil
import unittest
from chariot.storage import Storage


def resolve(path):
    return os.path.abspath(path)


class TestStorage(unittest.TestCase):

    def test_path(self):
        root = os.path.join(os.path.dirname(__file__), "../../")
        storage = Storage(root)
        correct_path = os.path.join(root, "data/raw")
        self.assertEqual(resolve(storage.data_path("raw")),
                         resolve(correct_path))

    def test_setup_data_dir(self):
        root = os.path.join(os.path.dirname(__file__), "./tmp_root")
        os.mkdir(root)

        storage = Storage.setup_data_dir(root)
        self.assertTrue(os.path.exists(storage.data_path("raw")))
        self.assertTrue(os.path.exists(storage.data_path("processed")))
        self.assertTrue(os.path.exists(storage.data_path("interim")))
        self.assertTrue(os.path.exists(storage.data_path("external")))
        shutil.rmtree(root)


if __name__ == "__main__":
    unittest.main()
