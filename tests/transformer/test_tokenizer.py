import unittest
from chariot.transformer.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):

    def test_ja_tokenize(self):
        tokenizer = Tokenizer(lang="ja")
        text = "日本語の形態素解析、マジ卍。"
        tokens = tokenizer.transform(text)

        for t in tokens:
            self.assertFalse(t.is_spacy)
            self.assertTrue(t.is_ja)
            self.assertTrue(t.surface)
            self.assertTrue(t.base_form)
            self.assertTrue(t.pos)
            self.assertTrue(t.tag)

    def test_en_tokenize(self):
        tokenizer = Tokenizer(lang="en")
        text = "Tom goes to a park that Mary is playing."
        tokens = tokenizer.transform(text)

        for t in tokens:
            self.assertTrue(t.is_spacy)
            self.assertFalse(t.is_ja)
            self.assertTrue(t.surface)
            self.assertTrue(t.base_form)
            self.assertTrue(t.pos)
            self.assertTrue(t.tag)

    def test_split_tokenize(self):
        tokenizer = Tokenizer(lang=None)
        text = "Tom goes to a park that Mary is playing."
        tokens = tokenizer.transform(text)

        for t in tokens:
            self.assertFalse(t.is_spacy)
            self.assertFalse(t.is_ja)
            self.assertTrue(t.surface)
            self.assertTrue(t.base_form)
            self.assertTrue(t.pos)
            self.assertTrue(t.tag)


if __name__ == "__main__":
    unittest.main()
