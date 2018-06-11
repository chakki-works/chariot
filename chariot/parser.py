from chariot.tokenizer import Tokenizer
from chariot.preprocessor.text import TextFilter, TextNormalizer
from chariot.preprocessor.token import TokenFilter, TokenNormalizer


class Parser():

    def __init__(self, tokenizer_or_lang,
                 text_preprocessors=(), token_preprocessors=()):
        self.text_preprocessors = text_preprocessors
        self.tokenizer = tokenizer_or_lang
        if isinstance(self.tokenizer, str):
            self.tokenizer = Tokenizer(tokenizer_or_lang)
        self.token_preprocessors = token_preprocessors

    @classmethod
    def create(cls, steps):
        text_ps = []
        token_ps = []
        tokenizer = None

        for p in steps:
            if isinstance(p, (TextFilter, TextNormalizer)):
                text_ps.append(p)
            elif isinstance(p, Tokenizer):
                tokenizer = p
            elif isinstance(p, (TokenFilter, TokenNormalizer)):
                token_ps.append(p)

        if tokenizer is None:
            raise Exception("Tokenizer should be included to parser.")

        return cls(tokenizer, text_ps, token_ps)

    def parse(self, text, return_surface=False):
        _text = text
        for tp in self.text_preprocessors:
            _text = tp.apply(_text)

        tokens = self.tokenizer.tokenize(_text)

        for tp in self.token_preprocessors:
            tokens = tp.apply(tokens)

        if return_surface:
            return [t.surface for t in tokens]
        else:
            return tokens
