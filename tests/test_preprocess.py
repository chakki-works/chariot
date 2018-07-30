import os
import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from chariot.preprocess import Preprocess


class TestPreprocess(unittest.TestCase):

    def test_preprocess(self):
        data, preprocess = self._make_preprocess()
        applied = preprocess.apply(data)

        self.assertEqual(len(applied), 3)
        for c in applied:
            self.assertTrue(c in ["label", "scaled_1", "scaled_2"])

    def test_save_load(self):
        data, preprocess = self._make_preprocess()
        path = os.path.join(os.path.dirname(__file__), "test_preprocess.tar.gz")
        preprocess.save(path)

        _preprocess = Preprocess.load(path)
        applied = _preprocess.apply(data)

        self.assertEqual(len(applied), 3)
        for c in applied:
            self.assertTrue(c in ["label", "scaled_1", "scaled_2"])
        os.remove(path)

    def _make_preprocess(self):
        data = {
            "label": np.random.uniform(size=100),
            "feature": np.random.uniform(size=100)
        }

        df = pd.DataFrame.from_dict(data)

        def column(name):
            return df[name].values.reshape(-1, 1)

        label_scaler = StandardScaler().fit(column("label"))
        feature_scaler_1 = StandardScaler().fit(column("feature"))
        feature_scaler_2 = MinMaxScaler().fit(column("feature"))

        p = Preprocess({
            "label": label_scaler,
            "feature": {
                "scaled_1": feature_scaler_1,
                "scaled_2": feature_scaler_2
            }
        })

        return df, p
