import unittest
import chariot.transformer.text as tfm


class TestTextTransformer(unittest.TestCase):

    def test_unicode_normalizer(self):
        text = "〒１１１－１１１１"
        normalized = tfm.UnicodeNormalizer().transform(text)
        self.assertEqual(normalized, "〒111-1111")

    def test_symbol_filter(self):
        text = "my symbol !! is replaced ? @^_^@"
        filtered = tfm.SymbolFilter(filters="!?@^_").transform(text)
        filtered = [w for w in filtered.split() if w.strip()]
        self.assertEqual(len(filtered), 4)
