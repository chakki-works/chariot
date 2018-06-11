import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import unittest
from sumbase.storage.csv_file import CsvFile


class TestCsvFile(unittest.TestCase):

    def test_read(self):
        path = os.path.join(os.path.dirname(__file__), "./sample_dataset.csv")
        csv = CsvFile(path, delimiter="\t")
        for line in csv.fetch():
            print(line)


if __name__ == "__main__":
    unittest.main()
