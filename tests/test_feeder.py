import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import unittest
from chariot.storage import Storage
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor
from chariot.preprocess import Preprocess
from chariot.feeder import Feeder


class TestFeeder(unittest.TestCase):

    def test_feed(self):
        path = os.path.join(os.path.dirname(__file__), "./")
        storage = Storage(path)
        df = storage.read("raw/corpus_multi.csv", delimiter="\t",
                          names=["label", "review", "comment"])

        # Make preprocessor
        preprocessor = Preprocessor(
                            tokenizer=ct.Tokenizer("en"),
                            text_transformers=[ct.text.UnicodeNormalizer()],
                            token_transformers=[ct.token.StopwordFilter("en")],
                            indexer=ct.Indexer(min_df=0, max_df=1.0))

        preprocessor.fit(df[["review", "comment"]])

        # Build preprocess
        prep = Preprocess({
            "review": preprocessor,
            "comment": preprocessor
        })

        # Apply preprocessed
        preprocessed = prep.apply(df)

        # Feed
        feeder = Feeder({"label": preprocessor.indexer.make_categorical(),
                         "review": preprocessor.indexer.make_padding(length=5)})

        adjusted = feeder.apply(preprocessed, ignore=("comment"))
        self.assertEqual(len(adjusted["label"][0]),
                         len(preprocessor.indexer.vocab))

        # Iterate
        for batch in feeder.iterate(preprocessed, batch_size=1, epoch=1, ignore=("comment")):
            self.assertEqual(len(batch), 2)
            self.assertEqual(len(batch["review"][0]), 5)


if __name__ == "__main__":
    unittest.main()
