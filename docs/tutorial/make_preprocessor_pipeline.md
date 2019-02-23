<h2 id="title">Make preprocessor pipeline</h2>
---

You can combine each preprocessor to make pipeline process. As the name pipeline indicates, it just same as the [scikit-learn Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

### Define a Pipeline

You can use `Preprocessor` to combine each preprocessor.

```py
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor


preprocessor = Preprocessor()
preprocessor\
    .stack(ct.text.UnicodeNormalizer())\
    .stack(ct.Tokenizer("en"))\
    .stack(ct.token.StopwordFilter("en"))\
    .stack(ct.Vocabulary(min_df=5, max_df=0.5))\
    .fit(train_data)

preprocessed = preprocessor.transform(data)
```

You can save & load the `Preprocessor`.

```
preprocessor.save("my_preprocessor.pkl")

loaded = Preprocessor.load("my_preprocessor.pkl")
```

It means you can pack & carry the preprocess by `.pkl` file.

### Make pipeline for dataset

When you want to apply distinctive preprocess for each column of a dataset, you can use `DatasetPreprocessor`.

```py
from chariot.dataset_preprocessor import DatasetPreprocessor
from chariot.transformer.formatter import Padding


dp = DatasetPreprocessor()
dp.process("review")\
    .by(ct.text.UnicodeNormalizer())\
    .by(ct.Tokenizer("en"))\
    .by(ct.token.StopwordFilter("en"))\
    .by(ct.Vocabulary(min_df=5, max_df=0.5))\
    .by(Padding(length=pad_length))\
    .fit(train_data["review"])
dp.process("polarity")\
    .by(ct.formatter.CategoricalLabel(num_class=3))


preprocessed = dp.preprocess(data)
```

You can save & load `DatasetPreprocessor` as preprocessor.

```py
dp.save("my_dataset_preprocessor.tar.gz")
loaded = DatasetPreprocessor.load("my_dataset_preprocessor.tar.gz")
```

Why you preprocess the data? Of course you want to train your model!  
Next [feed data to your model](train_model.md).
