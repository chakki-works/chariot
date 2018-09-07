<h2 id="title">Make a preprocessor pipeline</h2>
---


You can stack each preprocessor as preprocessor pipeline. As the name pipeline indicates, it just same as the [scikit-learn Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).

### Define a Pipeline

You can use `Preprocessor` to combine each preprocessors.

```py
import chariot.transformer as ct
from chariot.preprocessor import Preprocessor


preprocessor = Preprocessor(
                    tokenizer=ct.Tokenizer("en"),
                    text_transformers=[ct.text.UnicodeNormalizer()],
                    token_transformers=[ct.token.StopwordFilter("en")],
                    vocabulary=ct.Vocabulary(min_df=3, max_df=1.0))

preprocessed = preprocessor.fit_transform(text_document)
```

## Save and load a pipeline

You can save & load the `Preprocessor`.

```
preprocessor.save("/path/to/preprocessor_name.pkl")
preprocessor = Preprocess.load("/path/to/preprocessor_name.pkl")
```

It means you can deploy a serialized preprocessor and run it without its definition code!


### Apply individual pipeline

If you have multiple text column, you'll want to change preprocess column by column. `Preprocess` enables it.

```py
preprocess = Preprocess({
                "question": question_preprocessor,
                "document": document_preprocessor
             })
```

Or apply multiple preprocessors to one column.

```py
preprocess = Preprocess({
                "document": {
                    "word": word_preprocessor,
                    "charactor": char_preprocessor
                }
             })
```

You can save & load `Preprocess` as preprocessor.

```py
preprocess.save("/path/to/preprocess.tar.gz")
preprocess = Preprocess.load("/path/to/preprocess.tar.gz"
```

Now you prepare the data for your model. Then [feed it to your model](model_integration.md)!
