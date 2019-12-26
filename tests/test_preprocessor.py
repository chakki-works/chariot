import os
import unittest
from sklearn.externals import joblib
from chariot.storage import Storage
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor


class TestPreprocessor(unittest.TestCase):

    def test_dataframe(self):
        path = os.path.join(os.path.dirname(__file__), "./data")
        storage = Storage(path)
        df = storage.read("raw/corpus.csv", delimiter="\t",
                          names=["summary", "text"])

        preprocessor = Preprocessor(
                            tokenizer=ct.Tokenizer("ja"),
                            text_transformers=[ct.text.UnicodeNormalizer()],
                            vocabulary=ct.Vocabulary(vocab_size=50))

        preprocessor.fit(df[["summary", "text"]])
        joblib.dump(preprocessor, "test_preprocessor.pkl")

        preprocessor = joblib.load("test_preprocessor.pkl")
        transformed = preprocessor.transform(df)
        inversed = preprocessor.inverse_transform(transformed)

        for c in df.columns:
            for o, i in zip(df[c], inversed[c]):
                self.assertEqual(o, "".join(i))

        print(inversed)
        os.remove("test_preprocessor.pkl")

    def test_series(self):
        path = os.path.join(os.path.dirname(__file__), "./data")
        storage = Storage(path)
        df = storage.read("raw/corpus_multi.csv", delimiter="\t",
                          names=["label", "review", "comment"])

        preprocessor = Preprocessor(
                            tokenizer=ct.Tokenizer("en"),
                            text_transformers=[ct.text.UnicodeNormalizer()],
                            token_transformers=[ct.token.StopwordFilter("en")],
                            vocabulary=ct.Vocabulary(min_df=0, max_df=1.0))

        preprocessor.fit(df["review"])
        transformed = preprocessor.transform(df["comment"])
        self.assertEqual(len(transformed), 3)


if __name__ == "__main__":
    unittest.main()
