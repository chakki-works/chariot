import unittest
import chariot.transformer.text as tfm


class TestTextTransformer(unittest.TestCase):

    def test_unicode_normalizer(self):
        text = "〒１１１－１１１１"
        normalized = tfm.UnicodeNormalizer().transform([text])[0]
        self.assertEqual(normalized, "〒111-1111")

    def test_symbol_filter(self):
        text = "my symbol !! is replaced ? @^_^@"
        filtered = tfm.SymbolFilter(filters="!?@^_").transform([text])[0]
        filtered = [w for w in filtered.split() if w.strip()]
        self.assertEqual(len(filtered), 4)

    def test_lower_normalizer(self):
        text = "MY NAME is Joe"
        normalized = tfm.LowerNormalizer().transform([text])[0]
        self.assertEqual(normalized, "my name is joe")
