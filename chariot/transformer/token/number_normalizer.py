from chariot.transformer.token.base import TokenNormalizer


class NumberNormalizer(TokenNormalizer):

    def __init__(self, digit_character="0", copy=True):
        super().__init__(copy)
        self.digit_character = digit_character

    def apply(self, tokens):
        for t in tokens:
            is_number = False
            if t.is_spacy:
                if t._token.is_digit:
                    t.set_surface(self.digit_character)
                    is_number = True
            elif t.is_ja:
                # print(t.pos, t.tag)
                if (t.pos == "名詞" and t.tag == "数"):
                    t.set_surface(self.digit_character)
                    is_number = True

            escaped = t.surface.replace(".", "").replace(",", "")
            if not is_number and escaped.isdigit():
                t.set_surface(self.digit_character)

        return tokens
