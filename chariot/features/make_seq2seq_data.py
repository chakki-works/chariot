import os
import sys
import numbers
from collections import Counter
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from janome.tokenizer import Tokenizer
from src.utils.storage import Storage, File, CsvFile


def make_data(domain, version, in_column, out_column, min_df=1, stop_words=()):
    source_file = CsvFile.create("interim", domain, "formatted", version)
    if not source_file.exists():
        raise Exception("Source file does not exist.")

    # Make vocabulary
    #  load all data to array to prioritize speed than memory usage.
    vocab = Counter()
    tokenizer = Tokenizer(wakati=True)
    line_count = 0
    inputs = []
    outputs = []
    for element in source_file.fetch(delimiter="\t"):
        in_tokens = tokenizer.tokenize(element[in_column])
        out_tokens = tokenizer.tokenize(element[out_column])
        inputs.append(in_tokens)
        outputs.append(out_tokens)
        for t in (in_tokens + out_tokens):
            vocab[t] += 1
        line_count += 1

    # Limit the vocabulary size
    min_count = (min_df if isinstance(min_df, numbers.Integral)
                 else min_df * line_count)

    selected_vocab = []
    for term, count in vocab.most_common():
        if count <= min_count or term in stop_words:
            continue
        else:
            selected_vocab.append(term)

    print("Vocabulary size is {} (from {}).".format(
            len(selected_vocab), len(vocab)))
    # Add padding/unknown tag
    vocab = ["__PAD__", "__UNK__", "__SOS__", "__EOS__"] + selected_vocab

    # Write vocab file
    vocab_file = source_file.convert(data_dir_to="content",
                                     attribute_to="seq2seq", ext_to=".vocab")
    with open(vocab_file.path, mode="w", encoding="utf-8") as f:
        for v in vocab:
            f.write((v + "\n"))

    # Make tokenized file
    data_file = source_file.convert(data_dir_to="processed",
                                    attribute_to="seq2seq")
    unknown = vocab.index("__UNK__")
    start_index = vocab.index("__SOS__")
    end_index = vocab.index("__EOS__")
    get_index = lambda t: unknown if t not in vocab else vocab.index(t)
    with open(data_file.path, mode="w", encoding="utf-8") as f:
        for in_tokens, out_tokens in zip(inputs, outputs):
            input_indices = [str(get_index(t)) for t in in_tokens]
            output_indices = [str(get_index(t)) for t in out_tokens]
            output_indices = [str(start_index)] + output_indices + [str(end_index)]
            line = " ".join(input_indices) + "\t" + " ".join(output_indices)
            f.write(line + "\n")


if __name__ == "__main__":
    make_data(domain="cosme", version="20180524",
              in_column="description", out_column="catchphrase",
              min_df=3)
