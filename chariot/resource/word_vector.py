import numpy as np
from chariot.resource.data_file import DataFile


class WordVector(DataFile):

    def __init__(self, path, encoding="utf-8"):
        super().__init__(path, encoding)

    def load_embedding(self, vocab, progress=False):
        embedding = None

        initialized = False
        for line in self.fetch(progress):
            values = line.split()
            word = values[0]
            vector = values[1:]
            if not initialized:
                if word.isdigit() and\
                   (len(vector) == 1 and vector[0].isdigit()):
                    # Word2Vec format: First row is vocab_size & vector_size
                    # https://github.com/3Top/word2vec-api/issues/6#issuecomment-179339511
                    embedding_size = int(vector[0])
                    embedding = np.zeros((len(vocab), embedding_size))
                    initialized = True
                else:
                    embedding_size = len(vector)
                    embedding = np.zeros((len(vocab), embedding_size))
                    initialized = True

            if word in vocab:
                index = vocab.index(word)
                vector = np.asarray(values[1:], dtype="float32")
                embedding[index] = vector
        return embedding

    def load(self, progress=False):
        key_vector = {}
        vector_size = -1
        for line in self.fetch(progress):
            values = line.split()
            word = values[0]
            vector = values[1:]
            if vector_size < 0:
                if word.isdigit() and\
                   (len(vector) == 1 and vector[0].isdigit()):
                    vector_size = int(vector[0])
                else:
                    vector_size = len(vector)

            vector = np.asarray(values[1:], dtype="float32")
            key_vector[word] = vector
        return key_vector

    def load_model(self, binary):
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format(self.path, binary=binary)
        return model
