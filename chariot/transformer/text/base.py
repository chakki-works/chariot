from chariot.transformer.base_preprocessor import BasePreprocessor


class TextNormalizer(BasePreprocessor):

    def apply(self, text):
        raise Exception("You have to implements apply")


class TextFilter(BasePreprocessor):

    def apply(self, text):
        raise Exception("You have to implements apply")
