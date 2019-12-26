import os
import shutil
import unittest
from chariot.storage import Storage


def resolve(path):
    return os.path.abspath(path)


class TestStorage(unittest.TestCase):

    def test_path(self):
        root = os.path.join(os.path.dirname(__file__), "../../data")
        storage = Storage(root)
        correct_path = os.path.join(root, "raw")
        self.assertEqual(resolve(storage.path("raw")),
                         resolve(correct_path))

    def test_setup_data_dir(self):
        root = os.path.join(os.path.dirname(__file__), "./tmp_root")
        os.mkdir(root)

        storage = Storage.setup_data_dir(root)
        self.assertTrue(os.path.exists(storage.raw()))
        self.assertTrue(os.path.exists(storage.processed()))
        self.assertTrue(os.path.exists(storage.interim()))
        self.assertTrue(os.path.exists(storage.external()))
        shutil.rmtree(root)

    def test_download(self):
        url = "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
        root = os.path.join(os.path.dirname(__file__), "./data")
        storage = Storage(root)
        path = storage.download(url, "raw/image.png")
        self.assertTrue(os.path.exists(path))
        correct_path = os.path.join(root, "raw/image.png")
        self.assertEqual(resolve(path), resolve(correct_path))
        os.remove(path)


if __name__ == "__main__":
    unittest.main()
