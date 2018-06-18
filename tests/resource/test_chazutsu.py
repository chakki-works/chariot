import os
import sys
import unittest
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import chazutsu
from chariot.storage import Storage


class TestChazutsu(unittest.TestCase):

    def test_chazutsu(self):
        path = os.path.join(os.path.dirname(__file__), "../")
        storage = Storage(path)
        r = chazutsu.datasets.DUC2004().download(storage.path("raw"))
        dataset = storage.chazutsu(r).train_dataset
        print(dataset.to_dataframe().head(5))
        shutil.rmtree(dataset.file_root)


if __name__ == "__main__":
    unittest.main()
