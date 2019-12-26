import os
import shutil
from pathlib import Path
import gzip
from zipfile import ZipFile
import unittest
from unittest import mock
from chariot.storage import Storage
from chariot.resource.word_vector import WordVector
from chariot.transformer.vocabulary import Vocabulary


def mock_chakin_download(index, path):
    word2vec = [
        "apple 0 1 0 0 0",
        "banana 0 0 1 0 0",
        "cherry 0 0 0 1 0",
    ]
    file_name = "word2vec_dummy.txt.gz"
    file_path = Path(path).joinpath(file_name)
    with gzip.open(file_path, "wb")as f:
        f.write("\n".join(word2vec).encode("utf-8"))
    return file_path


def mock_chakin_download_zip(index, path):
    word2vec = [
        "apple 0 1 0 0 0",
        "banana 0 0 1 0 0",
        "cherry 0 0 0 1 0",
    ]
    file_path = Path(path).joinpath("word2vec_dummy2.txt")
    zip_path = Path(path).joinpath("word2vec_dummy2.zip")
    with file_path.open("wb") as f:
        f.write("\n".join(word2vec).encode("utf-8"))
    with ZipFile(zip_path, "w") as z:
        z.write(file_path,  "word2vec_dummy2/" + file_path.name)
    file_path.unlink()

    return zip_path


class TestChakin(unittest.TestCase):

    @mock.patch("chakin.download", side_effect=mock_chakin_download)
    def test_download(self, mock_download):
        path = os.path.join(os.path.dirname(__file__), "./data")
        storage = Storage(path)
        storage.chakin(lang="Japanese")
        vec_path = storage.chakin(name="fastText(ja)")
        self.assertTrue(os.path.exists(vec_path))
        os.remove(vec_path)

    @mock.patch("chakin.download", side_effect=mock_chakin_download_zip)
    def test_download_zip(self, mock_download):
        path = os.path.join(os.path.dirname(__file__), "./data")
        storage = Storage(path)
        vec_path = storage.chakin(name="fastText(ja)")
        print(vec_path)
        self.assertTrue(os.path.exists(vec_path))
        self.assertTrue(Path(vec_path).joinpath("word2vec_dummy2.txt").exists())
        shutil.rmtree(vec_path)

    def test_word_vector_resource(self):
        path = os.path.join(os.path.dirname(__file__), "./data")
        storage = Storage(path)

        vocab = Vocabulary()
        vocab.set(["you", "loaded", "word", "vector", "now"])

        vector_size = 50
        word2vec = [
            "you " + " ".join(["0"] * vector_size),
            "word " + " ".join(["1"] * vector_size),
            "now " + " ".join(["2"] * vector_size),
        ]
        word2vec_file = Path(storage.path("external/word2vec_dummyr.txt"))
        with word2vec_file.open(mode="w", encoding="utf-8") as f:
            f.write("\n".join(word2vec))

        wv = WordVector(word2vec_file)
        key_vector = wv.load()
        for k in key_vector:
            self.assertTrue(k in vocab.get())
            self.assertEqual(len(key_vector[k]), vector_size)

        embed = vocab.make_embedding(word2vec_file)
        self.assertEqual(embed.shape, (len(vocab.get()), vector_size))


if __name__ == "__main__":
    unittest.main()
