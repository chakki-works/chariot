from chariot.transformer.token.base import TokenNormalizer


class BaseFormNormalizer(TokenNormalizer):

    def __init__(self, copy=True):
        super().__init__(copy)

    def apply(self, tokens):
        for t in tokens:
            t.set_surface(t.base_form)

        return tokens
