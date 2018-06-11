class Tokenizer():

    def __init__(self, lang="en"):
        self.lang = lang
        self._tokenizer = None
        self.set_tokenizer()

    def set_tokenizer(self):
        import chariot.tokenizer.tokenizers as tk
        if self.lang == "ja":
            try:
                self.tokenizer = tk.MeCabTokenizer()
            except Exception as ex:
                self.tokenizer = tk.JanomeTokenizer()
        else:
            self.tokenizer = tk.SpacyTokenizer(self.lang)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)
