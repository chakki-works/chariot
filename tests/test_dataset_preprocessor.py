import os
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from chariot.dataset_preprocessor import DatasetPreprocessor
from chariot.storage import Storage
import chariot.transformer as ct
from chariot.transformer.formatter.base import BaseFormatter


class ScalingFormatter(BaseFormatter):

    def __init__(self, scale=2):
        super().__init__()
        self.scale = 2

    def transform(self, column):
        x = np.array(column) * self.scale
        return x


class TestDatasetPreprocessor(unittest.TestCase):

    def test_preprocess(self):
        data, dp = self._make_dp()
        preprocessed = dp.preprocess(data)
        self.assertEqual(len(preprocessed), 3)
        for c in preprocessed:
            self.assertTrue(c in ["label", "feature_1", "feature_2"])

        formatted = dp.format(preprocessed)
        self.assertEqual(len(formatted), 3)
        for c in formatted:
            self.assertTrue(c in ["label", "feature_1", "feature_3"])

        batch_size = 10
        for d in dp.iterate(preprocessed, batch_size=batch_size, epoch=1):
            for k in d:
                self.assertEqual(len(d[k]), batch_size)

    def test_save_load(self):
        data, dp = self._make_dp()
        path = os.path.join(os.path.dirname(__file__), "test_preprocess.tar.gz")
        dp.save(path)

        _dp = DatasetPreprocessor.load(path)
        preprocessed = _dp.preprocess(data)
        self.assertEqual(len(preprocessed), 3)
        for c in preprocessed:
            self.assertTrue(c in ["label", "feature_1", "feature_2"])

        os.remove(path)

    def _make_dp(self):
        data = {
            "label": np.random.uniform(size=100).reshape((-1, 1)),
            "feature": np.random.uniform(size=100).reshape((-1, 1))
        }

        label_scaler = StandardScaler().fit(data["label"])
        feature_scaler = StandardScaler().fit(data["feature"])
        feature_scaler_2 = MinMaxScaler().fit(data["feature"])

        dp = DatasetPreprocessor()
        dp.process("label").by(label_scaler)
        dp.process("feature")\
            .by(feature_scaler)\
            .by(feature_scaler_2).as_name("feature_1")
        dp.process("feature")\
            .by(feature_scaler).as_name("feature_2")\
            .by(ScalingFormatter()).as_name("feature_3")

        return data, dp

    def test_feed(self):
        path = os.path.join(os.path.dirname(__file__), "./data")
        storage = Storage(path)
        df = storage.read("raw/corpus_multi.csv", delimiter="\t",
                          names=["label", "review", "comment"])

        dp = DatasetPreprocessor()
        dp.process("review")\
            .by(ct.text.UnicodeNormalizer())\
            .by(ct.Tokenizer("en"))\
            .by(ct.token.StopwordFilter("en"))\
            .by(ct.Vocabulary(min_df=0, max_df=1.0))\
            .by(ct.formatter.Padding(length=5))\
            .fit(df.loc[:, ["review", "comment"]])
        dp.process("label")\
            .by(ct.formatter.CategoricalLabel(),
                reference=dp.process("review"))

        adjusted = dp(df).preprocess().format().processed
        self.assertEqual(len(adjusted["label"][0]),
                         dp.process("review").preprocessor.vocabulary.count)

        # Iterate
        for batch in dp(df).preprocess().iterate(batch_size=1, epoch=1):
            self.assertEqual(len(batch), 3)
            self.assertEqual(len(batch["review"][0]), 5)

            inversed = dp.inverse(batch)
            self.assertEqual(inversed["label"][0], np.argmax(batch["label"]))
            self.assertLessEqual(len(inversed["review"][0]), 5)
