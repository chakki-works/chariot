import os


class Vocabulary():

    def __init__(self, vocab=()):
        self._vocab = list(vocab)

    def vocab(self):
        if len(self._vocab) > 0:
            return self._vocab
        else:
            if not os.path.exists(self._vocab_path):
                raise Exception("Vocabulary file does not exist at {}.".format(
                    self._vocab_path
                ))

            with open(self._vocab_path, encoding="utf-8") as f:
                tokens = f.readlines()
                self._vocab = [t.strip() for t in tokens]
            return self._vocab
