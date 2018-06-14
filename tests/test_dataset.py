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

        feed = dataset.to_feed(field_transformers={"summary": preprocessor,
                                                   "text": preprocessor})

        s, t = feed.full()
        s_padded = s.adjust(padding=5)
        self.assertEqual(s_padded.shape, (3, 5))
        s_padded = s.adjust(padding=5, to_categorical=True)
        self.assertEqual(s_padded.shape, (3, 5, len(preprocessor.indexer.vocab)))

    def test_batch_iter(self):
        path = os.path.join(os.path.dirname(__file__), "./")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/corpus_multi.csv"), delimiter="\t")

        preprocessor = Preprocessor(
                            tokenizer=ct.Tokenizer("en"),
                            text_transformers=[ct.text.UnicodeNormalizer()],
                            token_transformers=[ct.token.StopwordFilter("en")],
                            indexer=ct.Indexer(min_df=0))

        dataset = Dataset(csv, ["label", "review", "comment"])
        preprocessor.fit(dataset.get("review", "comment"))

        feed = dataset.to_feed(field_transformers={
            "label": None,
            "review": preprocessor
        })

        count = 0
        batch_size = 2
        for labels, reviews in feed.batch_iter(batch_size=batch_size, epoch=2):
            _labels = labels.to_int_array()
            _reviews = reviews.adjust(padding=5, to_categorical=True)
            self.assertEqual(len(_labels), batch_size)
            self.assertEqual(_reviews.shape, (batch_size, 5,
                                              len(preprocessor.indexer.vocab)))
            count += 1

        self.assertEqual(count, 2)


if __name__ == "__main__":
    unittest.main()
