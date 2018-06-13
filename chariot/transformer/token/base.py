from chariot.transformer.base_preprocessor import BasePreprocessor


class TokenNormalizer(BasePreprocessor):

    def apply(self, tokens):
        raise Exception("You have to implements apply")


class TokenFilter(BasePreprocessor):

    def apply(self, tokens):
        raise Exception("You have to implements apply")
