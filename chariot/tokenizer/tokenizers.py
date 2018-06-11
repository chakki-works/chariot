from collections import namedtuple
from chariot.tokenizer.token import Token


class MeCabTokenizer():
    JanomeToken = namedtuple("JanomeToken", ("surface", "part_of_speech",
                                             "infl_type", "infl_form",
                                             "base_form", "reading",
                                             "phonetic"))

    def __init__(self):
        import MeCab
        self.tagger = MeCab.Tagger("-Ochasen")

    def tokenize(self, text):
        self.tagger.parse("")
        node = self.tagger.parseToNode(text)
        tokens = []
        while node:
            # Ignore BOS/EOS
            if node.surface:
                surface = node.surface
                features = node.feature.split(",")
                token = MeCabTokenizer.JanomeToken(
                            surface, ",".join(features[:4]),
                            features[4], features[5],
                            features[6], features[7],
                            features[8])
                token = Token(token, token_type="ja")
                tokens.append(token)
            node = node.next
        return tokens


class SpacyTokenizer():

    def __init__(self, lang):
        import spacy
        if lang in ["en", "de", "es", "pt", "fr", "it", "nl"]:
            nlp = spacy.load(lang)
        else:
            nlp = spacy.load("xx")

        self._nlp = nlp

    def tokenize(self, text):
        tokens = self._nlp(text)
        tokens = [Token(t) for t in tokens]
        return tokens


class JanomeTokenizer():

    def __init__(self):
        from janome.tokenizer import Tokenizer
        self.tokenizer = Tokenizer()

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)
        tokens = [Token(t, token_type="ja") for t in tokens]
        return tokens
