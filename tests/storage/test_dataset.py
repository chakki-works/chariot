import os
import unittest
import numpy as np
from src.utils.dataset import Seq2SeqDataset


def resolve(path):
    return os.path.abspath(path)


class TestDataset(unittest.TestCase):
    DATA_FILE = os.path.join(os.path.dirname(__file__), "sample_dataset.csv")

    def test_seq2seq_get(self):
        VOCAB_FILE = os.path.join(os.path.dirname(__file__),
                                  "sample_vocab.vocab")
        dataset = Seq2SeqDataset(path=self.DATA_FILE, vocab_path=VOCAB_FILE)
        X, y = dataset.get_Xy_padded(x_padding=20, y_padding=12)
        self.assertEqual(len(y), len(X))
        self.assertEqual(X[0][0], 4)

        self.assertEqual(X.shape, (len(X), 20))
        self.assertEqual(y.shape, (len(y), 12))

        print(dataset.inverse(X[0]))


if __name__ == "__main__":
    unittest.main()
