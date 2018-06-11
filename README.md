# chariot

Speedy data processing tool for NLP tasks

1. Data download & expansion powered by [chazutsu](https://github.com/chakki-works/chazutsu).
2. Tokenize data and make vocabulary powered by [spaCy](https://spacy.io/).
3. Prepare the pre-trained word vectors by [chakin](https://github.com/chakki-works/chakin)

Now, only you have to do is making the model!

The structure of data directory follows the [`Cookiecutter Data Science`](https://drivendata.github.io/cookiecutter-data-science/).

```
Project root
  └── data
       ├── external     <- Data from third party sources (ex. word vectors).
       ├── interim      <- Intermediate data that has been transformed.
       ├── processed    <- The final, canonical data sets for modeling.
       └── raw          <- The original, immutable data dump.
```

## Data download

Prepare the `requirements_d.txt` to manage the data dependency.

```
movie_review_data:polarity
your_original_data:v1 http://data/file/url/data.csv
```

## Tokenize data & make vocabulary

There is 2 type of preprocessors are prepared in `chariot`.

* TextPreprocessor
  * `TextNormalizer`: Normalize text (replace some character etc).
  * `TextFilter`: Filter the text (it means skip the line of text).
* TokenPreprocessor: Normalize of filter tokens.
  * `TokenNormalizer`: Normalize tokens (to lower, to original form etc).
  * `TokenFilter`: Filter tokens (extract only noun etc).

You can build the `Parser` from one `Tokenizer` and arbitrary number of TextPreprocessor and TokenPreprocessor.

```py
from chariot.parser import Parser, Tokenizer
from chariot.preprocessor.text import UnicodeNormalizer
from chariot.preprocessor.token import StopwardFilter


parser = Parser(steps=[
                UnicodeNormalizer(),
                Tokenizer(lang="en"),
                StopwardFilter()])

tokens = parser.parse("I like an aplle.")

```

After you tokenize the texts, then make vocabulary and indexing these.


```py
from chariot.storage import Storage
from chariot.storage.csv_file import CsvFile
from chariot.corpus import Corpus


storage = Storage(root="your/data/dir")
source_file = CsvFile(storage.data("raw/corpus.csv"), delimiter="\t")

corpus = Corpus.build(source_file, parser)


y_format_func = corpus.format_func(padding=5, to_categorical=True)
x_format_func = corpus.format_func(padding=10)

X, y = corpus.to_dataset(label_format_func=y_format_func,
                         feature_format_func=x_format_func)
```

## Prepare the pre-trained word vectors

```py
from chariot.storage import Storage
from chariot.corpus import Corpus
from chariot.wordvector import WordVectors


storage = Storage(root="your/data/dir")
corpus = Corpus.load(storage, "mycorpus")


# Download the word vectors (if it does not exist yet).
fast_text = WordVectors.load(storage, "fastText(en)")


embeddings = corpus.vocab_to_vector(fast_text)
```

