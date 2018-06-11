import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from chariot.storage import Storage
from chariot.storage.csv_file import CsvFile
from chariot.corpus import Corpus
from chariot.parser import Parser


class TestCorpus(unittest.TestCase):

    def test_fetch(self):
        path = os.path.join(os.path.dirname(__file__), "../")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/sample_dataset.csv"), delimiter="\t")
        corpus = Corpus(csv)

        for x, y in corpus.fetch():
            print("feature: {}, label: {}".format(x, y))

    def test_dataset(self):
        path = os.path.join(os.path.dirname(__file__), "../")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/sample_dataset.csv"), delimiter="\t")
        corpus = Corpus(csv)

        x_format_func = corpus.format_func(padding=6, bos=True, eos=True)
        y_format_func = corpus.format_func(padding=6, bos=True, eos=True,
                                           to_categorical=True)

        X, y = corpus.to_dataset(label_format_func=y_format_func,
                                 feature_format_func=x_format_func)

        self.assertEqual(X.shape, (5, 6))
        self.assertEqual(y.shape, (5, 6, len(corpus.vocab)))

    def test_dataset_multi(self):
        path = os.path.join(os.path.dirname(__file__), "../")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/sample_multi_dataset.csv"), delimiter="\t")
        corpus = Corpus(csv)

        x_format_func1 = corpus.format_func(padding=5)
        x_format_func2 = corpus.format_func(padding=6)
        Xs, y = corpus.to_dataset(feature_format_func=[x_format_func1,
                                                       x_format_func2],
                                  word_sequence=True)

        self.assertEqual(len(Xs), 2)
        self.assertEqual(Xs[0].shape, (10, 5))
        self.assertEqual(Xs[1].shape, (10, 6))
        self.assertEqual(y.shape, (10, 1))

    def test_build(self):
        path = os.path.join(os.path.dirname(__file__), "../")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/corpus.csv"), delimiter="\t")
        parser = Parser("ja")
        corpus = Corpus.build(csv, parser)

    def test_build_multi(self):
        path = os.path.join(os.path.dirname(__file__), "../")
        storage = Storage(path)
        csv = CsvFile(storage.data("raw/corpus_multi.csv"), delimiter="\t")
        parser = Parser("en")
        corpus = Corpus.build(csv, parser, target_columns=(1, 2))


if __name__ == "__main__":
    unittest.main()
