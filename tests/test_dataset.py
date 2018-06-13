import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from chariot.storage import Storage
from chariot.storage.csv_file import CsvFile
from chariot.dataset import Dataset


class TestDataset(unittest.TestCase):

    def test_dataset(self):
        path = os.path.join(os.path.dirname(__file__), "./")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/sample_dataset.csv"), delimiter="\t")
        dataset = Dataset(csv, ["summary", "text"])
        for d in dataset():
            print(d)

        dumped = dataset.all("summary")
        print(dumped)


if __name__ == "__main__":
    unittest.main()
