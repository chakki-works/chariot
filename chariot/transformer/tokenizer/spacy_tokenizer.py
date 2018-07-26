from chariot.transformer.tokenizer.token import Token


class SpacyTokenizer():

    def __init__(self, lang):
        disables = ["textcat", "ner", "parser"]
        import spacy
        if lang in ["en", "de", "es", "pt", "fr", "it", "nl"]:
            nlp = spacy.load(lang, disable=disables)
        else:
            nlp = spacy.load("xx", disable=disables)

        self._nlp = nlp

    def tokenize(self, text):
        tokens = self._nlp(text)
        return [Token(t) for t in tokens]
