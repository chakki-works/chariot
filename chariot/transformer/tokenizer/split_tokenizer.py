from chariot.transformer.tokenizer.token import Token


class SplitTokenizer():

    def __init__(self):
        pass

    def tokenize(self, text):
        return [Token(t, token_type="-") for t in text.split()]
