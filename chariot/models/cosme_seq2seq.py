from janome.tokenizer import Tokenizer
from src.features.format_cosme import CosmeFormatter
from src.models.seq2seq import SimpleSeq2Seq
from src.utils.dataset import Seq2SeqDataset


class CosmeSeq2SeqAPI():

    def __init__(self, cosme_data_file, embedding_size=-1, hidden_size=-1):
        self.dataset = Seq2SeqDataset(cosme_data_file.path)
        self.formatter = CosmeFormatter()
        self._model = None
        self._tokenizer = Tokenizer(wakati=True)
        if embedding_size > 0 and hidden_size > 0:
            self.make_model(embedding_size, hidden_size)
        else:
            self.make_model_deterministic()

    def make_model(self, embedding_size, hidden_size):
        self._model = SimpleSeq2Seq(vocab_size=len(self.dataset.vocab),
                                    embedding_size=embedding_size,
                                    hidden_size=hidden_size)
        self._model.build()

    def make_model_deterministic(self):
        self.make_model(embedding_size=200, hidden_size=200)

    def load(self, model_path):
        self._model.load(model_path)

    @property
    def model(self):
        return self._model.model

    @property
    def vocab_size(self):
        return self._model.vocab_size

    @property
    def embedding_size(self):
        return self._model.embedding_size

    @property
    def hidden_size(self):
        return self.hidden_size

    def generate(self, input_text, count=5):
        formatted = self.formatter.format(input_text)
        if len(formatted) == 0:
            raise Exception("Input text does not appropriate.")

        tokenized = self._tokenizer.tokenize(formatted)
        indices = self.dataset.to_indices(tokenized)
        generateds = []
        for i in range(count):
            generated = self._model.inference(indices,
                                              self.dataset.sos,
                                              self.dataset.eos)
            text = self.dataset.inverse(generated)
            generateds.append(text)

        return generateds
