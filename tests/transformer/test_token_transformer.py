import unittest
from chariot.transformer.tokenizer import Tokenizer
import chariot.transformer.token as tfm


class TestTokenTransformer(unittest.TestCase):

    def test_stopword_ja(self):
        tokenizer = Tokenizer(lang="ja")
        text = "わたしの形態素解析、マジ卍。"
        tokens = tokenizer.transform(text)
        filtered = tfm.StopwordFilter(lang="ja").transform(tokens)
        self.assertTrue(1, len(tokens) - len(filtered))

    def test_en_tokenize(self):
        tokenizer = Tokenizer(lang="en")
        text = "Tom goes to a park that Mary is playing."
        tokens = tokenizer.transform(text)
        filtered = tfm.StopwordFilter(lang="en").transform(tokens)
        self.assertTrue(4, len(tokens) - len(filtered))

    def test_baseform_normalizer(self):
        tokenizer = Tokenizer(lang="en")
        text = "goes playing"
        tokens = tokenizer.transform(text)
        filtered = tfm.BaseFormNormalizer().transform(tokens)
        self.assertTrue("go", filtered[0])
        self.assertTrue("play", filtered[1])
