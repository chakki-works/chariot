import unittest
from chariot.transformer.tokenizer import Tokenizer
from chariot.transformer.vocabulary import Vocabulary


class TestVocabulary(unittest.TestCase):

    def test_setter(self):
        vocab = Vocabulary(padding="_pad_", unknown="_unk_",
                           begin_of_sequence="_bos_", end_of_sequence="_eos_",)
        words = ["you", "are", "making", "the", "vocabulary"]
        vocab.set(words)

        vocab_size = len(words) + len(["_pad_", "_unk_", "_bos", "_eos"])
        self.assertEqual(len(vocab.get()), vocab_size)
        self.assertEqual(vocab.count, vocab_size)
        self.assertEqual(vocab.pad, vocab.get().index("_pad_"))
        self.assertEqual(vocab.unk, vocab.get().index("_unk_"))
        self.assertEqual(vocab.bos, vocab.get().index("_bos_"))
        self.assertEqual(vocab.eos, vocab.get().index("_eos_"))

    def test_setter_token(self):
        vocab = Vocabulary()
        text = "you are making the vocabulary"
        words = Tokenizer(lang="en").transform([text])[0]
        vocab.set(words)

        vocab_size = len(words) + len(["_pad_", "_unk_", "_bos", "_eos"])
        self.assertEqual(len(vocab.get()), vocab_size)
        print(vocab.get())

    def test_fit_transform(self):
        vocab = Vocabulary(padding="_pad_", unknown="_unk_",
                           begin_of_sequence="_bos_", end_of_sequence="_eos_",
                           min_df=1)

        doc = [
            ["you", "are", "reading", "the", "book"],
            ["I", "don't", "know", "its", "title"],
        ]

        vocab.fit(doc)
        print(vocab.get())
        text = ["you", "know", "book", "title"]
        indexed = vocab.transform([text])
        inversed = vocab.inverse_transform(indexed)[0]
        self.assertEqual(tuple(text), tuple(inversed))

        text = ["you", "know", "my", "title"]
        indexed = vocab.transform([text])
        inversed = vocab.inverse_transform(indexed)[0]
        self.assertEqual(inversed[2], "_unk_")

    def test_ignore_blank(self):
        vocab = Vocabulary(padding="_pad_", unknown="_unk_",
                           begin_of_sequence="_bos_", end_of_sequence="_eos_",
                           min_df=1, ignore_blank=True)

        doc = [
            ["you", "", "reading", "the", "book"],
            ["I", "don't", "\n", "its", "title"],
        ]
        words = []
        for s in doc:
            words += s

        vocab.fit(doc)
        length = len(words) - len(["", "\n"]) + len(["pad", "unk", "bos", "eos"])
        self.assertEqual(len(vocab.get()), length)

        vocab.ignore_blank=False
        vocab.fit(doc)
        self.assertEqual(len(vocab.get()), length + 1) 


if __name__ == "__main__":
    unittest.main()
