import os
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from chariot.dataset_preprocessor import DatasetPreprocessor
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
        self.assertEqual(len(preprocessed), 2)
        for c in preprocessed:
            self.assertTrue(c in ["label", "feature"])

        formatted = dp.format(preprocessed)
        self.assertEqual(len(formatted), 2)
        for c in formatted:
            self.assertTrue(c in ["label", "feature"])

    def xtest_save_load(self):
        data, preprocess = self._make_preprocess()
        path = os.path.join(os.path.dirname(__file__), "test_preprocess.tar.gz")
        preprocess.save(path)

        _preprocess = DatasetPreprocessor.load(path)
        applied = _preprocess.transform(data)

        self.assertEqual(len(applied), 2)
        for c in applied:
            self.assertTrue(c in ["label", "feature"])
        os.remove(path)

    def _make_dp(self):
        data = {
            "label": np.random.uniform(size=100),
            "feature": np.random.uniform(size=100)
        }

        df = pd.DataFrame.from_dict(data)

        def column(name):
            return df[name].values.reshape(-1, 1)

        label_scaler = StandardScaler().fit(column("label"))
        feature_scaler = StandardScaler().fit(column("feature"))

        dp = DatasetPreprocessor()
        dp.field("label").bind(label_scaler)
        dp.field("feature").bind(feature_scaler).bind(ScalingFormatter())

        return df, dp
