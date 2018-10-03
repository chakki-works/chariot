class Token():

    def __init__(self, token, token_type="spaCy"):
        self._token = token
        self.token_type = token_type
        self.__surface = ""

    def set_surface(self, surface):
        self.__surface = surface

    @property
    def is_spacy(self):
        return self.token_type == "spaCy"

    @property
    def is_ja(self):
        return self.token_type == "ja"

    @property
    def surface(self):
        if self.__surface:
            return self.__surface
        elif self.token_type == "spaCy":
            return self._token.text
        elif self.token_type == "ja":
            return self._token.surface
        else:
            return self._token

    @property
    def base_form(self):
        if self.token_type == "spaCy":
            return self._token.lemma_
        elif self.token_type == "ja":
            return self._token.base_form
        else:
            return "-"

    @property
    def pos(self):
        if self.token_type == "spaCy":
            return self._token.pos_
        elif self.token_type == "ja":
            pos = self._token.part_of_speech.split(",")[0]
            return pos
        else:
            return "-"

    @property
    def tag(self):
        if self.token_type == "spaCy":
            return self._token.tag_
        elif self.token_type == "ja":
            tag = self._token.part_of_speech.split(",")[1]
            return tag
        else:
            return "-"

    def __repr__(self):
        return "<{}:{}>".format(self.surface, self.pos)

    def __reduce_ex__(self, proto):
        return type(self), (self._token, self.token_type)
