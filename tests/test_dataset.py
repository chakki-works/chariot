import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from chariot.storage import Storage
from chariot.storage.csv_file import CsvFile
from chariot.dataset import Dataset
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor
from chariot.dataset import TransformedDataset


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

    def test_batch(self):
        path = os.path.join(os.path.dirname(__file__), "./")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/corpus.csv"), delimiter="\t")

        preprocessor = Preprocessor(
                            tokenizer=ct.Tokenizer("ja"),
                            text_transformers=[ct.text.UnicodeNormalizer()],
                            indexer=ct.Indexer(min_df=0))

        dataset = Dataset(csv, ["summary", "text"])
        preprocessor.fit(dataset.get())

        batch = dataset.batch(field_transformers={"summary": preprocessor,
                                                  "text": preprocessor})

        s, t = batch.full()
        s_padded = s.adjust(padding=5)
        self.assertEqual(s_padded.shape, (3, 5))
        s_padded = s.adjust(padding=5, to_categorical=True)
        self.assertEqual(s_padded.shape, (3, 5, len(preprocessor.indexer.vocab)))


if __name__ == "__main__":
    unittest.main()
