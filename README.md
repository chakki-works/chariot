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

## Install

```
pip install chariot
```

## Data download

You can download various dataset by using [chazutsu](https://github.com/chakki-works/chazutsu).  
And read its data to `chariot.dataset`.

```py
import chazutsu
from chariot.storage import Storage


storage = Storage("your/data/root")
r = chazutsu.datasets.IMDB().download(storage.path("raw"))

dataset = storage.chazutsu(r).train_dataset
dataset.to_dataframe().head(5)
```

Then

```
   polarity  rating                                             review
0         0       3  You'd think the first landing on the Moon woul...
1         1       9  I took a flyer in renting this movie but I got...
2         1      10  Sometimes I just want to laugh. Don't you? No ...
3         0       2  I knew it wasn't gunna work out between me and...
4         0       2  Sometimes I rest my head and think about the r...
```


## Preprocess the NLP data

All preprocessors are defined at `chariot.transformer`.  
Transformers are implemented following to the scikit-learn transformer manner.  Thanks to that, you can chain & save preprocessors easily.


```py
from sklearn.externals import joblib
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor


preprocessor = Preprocessor(
                  tokenizer=ct.Tokenizer("en"),
                  text_transformers=[ct.text.UnicodeNormalizer()],
                  token_transformers=[ct.token.StopwordFilter("en")],
                  indexer=ct.Indexer())

preprocessor.fit(your_dataset)
joblib.dump(preprocessor, "preprocessor.pkl")  # Save

preprocessor = joblib.load("preprocessor.pkl")  # Load
```

It means you don't need code of preprocessors in your inference (predict) server.

There is 5 type of transformers for preprocessors.

* TextPreprocessor
  * Preprocess the text before tokenization.
  * `TextNormalizer`: Normalize text (replace some character etc).
  * `TextFilter`: Filter the text (delete some span in text stc).
* Tokenizer
  * Tokenize the texts.
  * It powered by [spaCy](https://spacy.io/) and you can choose [MeCab](https://github.com/taku910/mecab) or [Janome](https://github.com/mocobeta/janome) for Japanese.
* TokenPreprocessor
  * Normalize/Filter the tokens after tokenization.
  * `TokenNormalizer`: Normalize tokens (to lower, to original form etc).
  * `TokenFilter`: Filter tokens (extract only noun etc).
* Indexer
  * Make vocabulary and convert tokens to indices.
* Adjuster
  * After the data is converted to indices (=int array), padding sequence etc.

You can save the preprocessed result to disk.

```py
from chariot.dataset import Dataset


dataset = Dataset(csv_file, ["label", "review", "comment"])

# Save indexed
indexed = dataset.save_transformed("token_to_indexed", {
            "label": None,
            "review": preprocessor
          })

```

The transformers are also saved with transformed data. For that reason you can inverse-transform the data after loading the dataset. 

```py
# Load indexed data
indexed = TransformedDataset.load(original_csv_file, "token_to_indexed")

words = indexed.field_transformers["review"].inverse_transform(indexed.get("review"))
```

## Feed the data to your model

`chariot` supports the feature for feeding the data to the model.

```py
sentiment_dataset = Dataset(csv_file, ["label", "review"])
preprocessor.fit(sentiment_dataset.get("review"))

feed = sentiment_dataset.to_feed(field_transformers={
    "label": None,
    "review": preprocessor
})

for labels, reviews in feed.batch_iter(batch_size=32, epoch=10):
    y = labels.to_int_array()
    X = reviews.adjust(padding=5, to_categorical=True)

    your_model.train(X, y)
```

## Prepare the pre-trained word vectors

You can download the pre-trained word vectors by [chakin](https://github.com/chakki-works/chakin).  
And use these easily.

```py
from chariot.storage import Storage
from chariot.transformer.indexer import Indexer

# Download word vector
storage = Storage("your/data/root")
storage.chakin(name="GloVe.6B.50d")

# Make embedding matrix
indexer = Indexer()
indexer.load_vocab("your/vocab/file/path")
embed = indexer.make_embedding(storage.path("external/glove.6B.50d.txt"))
print(embed.shape)  # len(indexer.vocab) x 50 matrix
```
