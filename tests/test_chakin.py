import os
import unittest
from chariot.storage import Storage
from chariot.transformer.indexer import Indexer


class TestChakin(unittest.TestCase):

    def test_download(self):
        path = os.path.join(os.path.dirname(__file__), "./")
        storage = Storage(path)
        storage.chakin(lang="Japanese")

        #storage.chakin(name="fastText(ja)")

    def test_word_vector_resource(self):
        path = os.path.join(os.path.dirname(__file__), "./")
        storage = Storage(path)

        indexer = Indexer()
        indexer.load_vocab(["you", "loaded", "word", "vector", "now"])
        embed = indexer.make_embedding(storage.data_path("external/glove.6B.50d.txt"))

        self.assertEqual(embed.shape, (len(indexer.vocab), 50))


if __name__ == "__main__":
    unittest.main()
