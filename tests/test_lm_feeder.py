import unittest
import numpy as np
import pandas as pd
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor
from chariot.feeder import LanguageModelFeeder


TEXT = """
A chariot is a type of carriage driven by a charioteer, usually using horses[a] to provide rapid motive power. Chariots were used by armies as transport or mobile archery platforms, for hunting or for racing, and as a conveniently fast way to travel for many ancient people.
The word "chariot" comes from the Latin term carrus, a loanword from Gaulish. A chariot of war or one used in military parades was called a car. In ancient Rome and some other ancient Mediterranean civilizations, a biga required two horses, a triga three, and a quadriga four.
"""

class TestLanguageModelFeeder(unittest.TestCase):

    def _make_corpus(self):
        return pd.DataFrame.from_dict({"sentence": [TEXT]})

    def test_feed(self):
        df = self._make_corpus()
        # Make preprocessor
        preprocessor = Preprocessor(
                            tokenizer=ct.Tokenizer("en"),
                            text_transformers=[ct.text.UnicodeNormalizer()],
                            token_transformers=[ct.token.StopwordFilter("en")],
                            vocabulary=ct.Vocabulary(vocab_size=30))

        preprocessed = preprocessor.fit_transform(df)
        feeder = LanguageModelFeeder({"sentence": ct.formatter.ShiftGenerator()})

        # Iterate
        b_len = 2
        s_len = 6
        for d, t in feeder.iterate(preprocessed, batch_size=b_len,
                                   sequence_length=s_len, epoch=2):
            self.assertEqual(d.shape, (s_len, b_len))
            self.assertEqual(t.shape, (s_len, b_len))

    def test_feed_sequential(self):
        feeder = LanguageModelFeeder({"sentence": ct.formatter.ShiftGenerator()})
        content = np.arange(20).reshape(1, -1)
        data = {"sentence": content}

        # Iterate
        b_len = 2
        s_len = 3
        batches = content.reshape((b_len, -1)).T
        index = 0
        for d, t in feeder.iterate(data, batch_size=b_len,
                                   sequence_length=s_len, epoch=1):
            self.assertEqual(d.tolist(), batches[index:index+s_len].tolist())
            self.assertEqual(t.tolist(), batches[index+1:index+1+s_len].tolist())
            index += s_len

    def test_feed_batch(self):
        feeder = LanguageModelFeeder({"sentence": ct.formatter.ShiftGenerator()})
        content = np.arange(122).reshape(1, -1)
        data = {"sentence": content}

        # Iterate
        b_len = 2
        s_len = 6
        batches = content.reshape((b_len, -1)).T

        index = 0
        epoch = 3
        epoch_count = 1
        for d, t, done in feeder.iterate(data, batch_size=b_len,
                                         sequence_length=s_len, epoch=epoch,
                                         sequencial=False, output_epoch_end=True):

            self.assertEqual(d.tolist(), batches[index:index+s_len].T.tolist())
            self.assertEqual(t.tolist(), batches[index+1:index+s_len+1].T.tolist())

            index += s_len
            if index + s_len >= len(batches):
                index = 0
            if done:
                epoch_count += 1

        self.assertEqual(epoch_count, epoch)


if __name__ == "__main__":
    unittest.main()
