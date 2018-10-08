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

    def test_feed_content_seq(self):
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

    def test_feed_content(self):
        feeder = LanguageModelFeeder({"sentence": ct.formatter.ShiftGenerator()})
        content = np.arange(120).reshape(1, -1)
        data = {"sentence": content}

        # Iterate
        b_len = 2
        s_len = 6
        batches = content.reshape((b_len, -1)).T
        """
        batches will be
       [[ 0 10]
        [ 1 11]
        [ 2 12]
        [ 3 13]
        [ 4 14]
        ...

        if seq = 3,
       [[ 0 10]
        [ 1 11]
        [ 2 12]]

        [ 3 13]
        
        to feed the model, transpose above.
        [[0 1 2]
         [10 11 12]]
        [3,
         13]
        """
        index = 0
        s_len_i = 1
        epoch = 3
        for d, t in feeder.iterate(data, batch_size=b_len,
                                   sequence_length=s_len, epoch=epoch,
                                   sequencial=False):
            #print("{} => {}".format(d.reshape((b_len, -1)), t.reshape((b_len, -1))))
            self.assertEqual(d.tolist(), batches[index:index+s_len_i].T.tolist())
            self.assertEqual(t.tolist(), batches[index+s_len_i].tolist())
            s_len_i += 1
            if s_len_i > s_len:
                s_len_i = 1
                index += s_len
            if index + s_len >= len(batches):
                index = 0
                s_len_i = 1


if __name__ == "__main__":
    unittest.main()
