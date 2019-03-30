import unittest
from chariot.transformer.tokenizer import Tokenizer
import chariot.transformer.token as tfm


class TestTokenTransformer(unittest.TestCase):

    def test_stopword_ja(self):
        tokenizer = Tokenizer(lang="ja")
        text = "わたしの形態素解析、マジ卍。"
        tokens = tokenizer.transform([text])
        filtered = tfm.StopwordFilter(lang="ja").transform(tokens)
        self.assertTrue(1, len(tokens) - len(filtered))

    def test_word_frequency_filter_ja(self):
        tokenizer = Tokenizer(lang="ja")
        text = "わたしの形態素解析は楽しい。わたしはまだまだ。"
        tokens = tokenizer.transform([text])
        filtered = tfm.WordFrequencyFilter(n=1, min_freq=1).fit_transform(tokens)[0]
        self.assertTrue(4, len(filtered))

    def test_word_frequency_filter_en(self):
        tokenizer = Tokenizer(lang="en")
        text = "Tom goes to a park that Mary is playing. Tom and Mary is playing tennis in the park."
        tokens = tokenizer.transform([text])
        filtered = tfm.WordFrequencyFilter(n=3, min_freq=1).fit_transform(tokens)[0]
        self.assertTrue(6, len(filtered))

    def test_en_tokenize(self):
        tokenizer = Tokenizer(lang="en")
        text = "Tom goes to a park that Mary is playing."
        tokens = tokenizer.transform([text])
        filtered = tfm.StopwordFilter(lang="en").transform(tokens)
        self.assertTrue(4, len(tokens) - len(filtered))

    def test_baseform_normalizer(self):
        tokenizer = Tokenizer(lang="en")
        text = "goes playing"
        tokens = tokenizer.transform([text])
        normalized = tfm.BaseFormNormalizer().transform(tokens)[0]
        self.assertTrue("go", normalized[0])
        self.assertTrue("play", normalized[1])

    def test_baseform_normalizer(self):
        tokenizer = Tokenizer(lang="en")
        text = "five players of 3 state on 1,000 location"
        tokens = tokenizer.transform([text])
        normalized = tfm.NumberNormalizer().transform(tokens)[0]
        self.assertEqual(2, len([t for t in normalized if t.surface == "0"]))

        tokenizer = Tokenizer(lang="ja")
        text = "百に一つの場所に2人の人がいる"
        tokens = tokenizer.transform([text])
        normalized = tfm.NumberNormalizer().transform(tokens)[0]
        self.assertEqual(2, len([t for t in normalized if t.surface == "0"]))
