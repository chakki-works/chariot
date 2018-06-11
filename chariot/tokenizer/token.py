class Token():

    def __init__(self, token, token_type="spaCy"):
        self._token = token
        self.token_type = token_type

    @property
    def surface(self):
        if self.token_type == "spaCy":
            return self._token.text
        else:
            return self._token.surface

    @property
    def base_form(self):
        if self.token_type == "spaCy":
            return self._token.lemma_
        else:
            return self._token.base_form

    @property
    def pos(self):
        if self.token_type == "spaCy":
            return self._token.pos_
        else:
            return self._token.part_of_speech[0]

    @property
    def tag(self):
        if self.token_type == "spaCy":
            return self._token.tag_
        else:
            return self._token.part_of_speech[1]

    def __repr__(self):
        return "<{}:{}>".format(self.surface, self.pos)
