import os
import sys
import unittest
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from chariot.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):

    def test_ja_tokenize(self):
        tokenizer = Tokenizer(lang="ja")
        text = "日本語の形態素解析、マジ卍。"
        tokens = tokenizer.tokenize(text)
        print(tokens)

    def test_en_tokenize(self):
        tokenizer = Tokenizer(lang="en")
        text = "Tom goes to a park that Mary is playing."
        tokens = tokenizer.tokenize(text)
        print(tokens)


if __name__ == "__main__":
    unittest.main()
