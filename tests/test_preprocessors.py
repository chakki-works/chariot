import os
import sys
import unittest
from sklearn.externals import joblib
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from chariot.storage import Storage
from chariot.storage.csv_file import CsvFile
from chariot.dataset import Dataset
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor


class TestPreprocessors(unittest.TestCase):

    def xtest_preprocess(self):
        path = os.path.join(os.path.dirname(__file__), "./")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/corpus.csv"), delimiter="\t")

        preprocessor = Preprocessor(
                            tokenizer=ct.Tokenizer("ja"),
                            text_transformers=[ct.text.UnicodeNormalizer()],
                            indexer=ct.Indexer(min_df=0))

        dataset = Dataset(csv, ["summary", "text"])
        preprocessor.fit(dataset.get())
        joblib.dump(preprocessor, "test_preprocessor.pkl")

        preprocessor = joblib.load("test_preprocessor.pkl")
        transformed = preprocessor.transform(dataset.get())
        inversed = preprocessor.inverse_transform(transformed)

        original = dataset.get()
        for k in original:
            for o, i in zip(original[k], inversed[k]):
                self.assertEqual(o, "".join(i))

        print(original)
        os.remove("test_preprocessor.pkl")

    def test_indexed_dataset(self):
        path = os.path.join(os.path.dirname(__file__), "./")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/corpus_multi.csv"), delimiter="\t")

        preprocessor = Preprocessor(
                            tokenizer=ct.Tokenizer("en"),
                            text_transformers=[ct.text.UnicodeNormalizer()],
                            token_transformers=[ct.token.StopwordFilter("en")],
                            indexer=ct.Indexer(min_df=0))

        dataset = Dataset(csv, ["label", "review", "comment"])
        preprocessor.fit(dataset.get(["review", "comment"]))

        indexed = dataset.save_indexed({
            "label": None,
            "review": preprocessor
        })
        print(indexed.get())
        print(preprocessor.inverse_transform(indexed.get("review")))


if __name__ == "__main__":
    unittest.main()
