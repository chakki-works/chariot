import os
import sys
import argparse
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from src.utils.storage import Storage, CsvFile
from src.models.cosme_seq2seq import CosmeSeq2SeqAPI
from src.models.seq2seq import Seq2SeqTrainer


def train(data_file):
    cosme_model = CosmeSeq2SeqAPI(data_file,
                                  embedding_size=200, hidden_size=200)
    input_size = 120
    output_size = 30
    trainer = Seq2SeqTrainer(cosme_model)
    trainer.train(cosme_model.dataset,
                  x_padding=input_size, y_padding=output_size)


def predict(data_file, count=10):
    # Use deterministic model. Check the coincide of train parameter
    storage = Storage()
    cosme_model = CosmeSeq2SeqAPI(data_file)
    model_path = storage.model(cosme_model)
    cosme_model.load(model_path)

    # Test description
    desc = "優れた洗浄力で、肌に必要な潤いは守りながら、汚れをすっきりオフ。洗い上がりはつっぱり感もなく、しなやかに肌を整えます。"

    generateds = cosme_model.generate(desc)
    for g in generateds:
        print(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seq2Seq experiment")
    parser.add_argument("--predict", action="store_true",
                        help="predict by the model")
    args = parser.parse_args()

    data_file = CsvFile.create("processed", "cosme", "seq2seq", "20180524")
    if args.predict:
        predict(data_file)
    else:
        train(data_file)
