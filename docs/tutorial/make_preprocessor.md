<h2 id="title">Make preprocessor</h2>
---

There are 6 kinds of preprocessor in `chariot`.

* TextPreprocessor
* Tokenizer
* TokenPreprocessor
* Vocabulary
* Formatter
* Generator

Each preprocessor is defined as [scikit-learn `Transformer`](http://scikit-learn.org/stable/modules/generated/sklearn.base.TransformerMixin.html). Because of this, these preprocessors locate at `chariot.transformer`.

You can initialize parameters of preprocessor by `fit`, and apply preprocess by `transform`.

### Text preprocessor

The role of the `Text preprocessor` is arranging text before tokenization.

```py
import chariot.transformer as ct


text = "Hey! you preprocess text now :)"
preprocessed = ct.text.SymbolFilter().transform(text)
```

```
> Hey you preprocess text now
```

### Tokenizer

The role of the `Tokenizer` is tokenizing text.

```py
import chariot.transformer as ct


text = "Hey you preprocess text now"
tokens = ct.Tokenizer(lang="en").transform(text)
```

```
> [<Hey:INTJ>, <you:PRON>, <preprocess:ADJ>, <text:NOUN>, <now:ADV>]
```


When tokenize a text, chariot use [`spaCy`](https://github.com/explosion/spaCy) mainly. You can specify language by `lang` parameter. But if you want to tokenize Japanese text, you have to prepare the [`Janome`](http://mocobeta.github.io/janome/) or [`MeCab`](http://taku910.github.io/mecab/) since spaCy does not support Japanese well.

### Token preprocessor

The role of the `Token preprocessor` is filter/normalize tokens before building vocabulary.

```py
import chariot.transformer as ct


text = "Hey you preprocess text now"
tokens = ct.Tokenizer(lang="en").transform(text)
filtered = ct.token.StopwordFilter(lang="en").transform(tokens)
```

```
> [<Hey:INTJ>, <preprocess:ADJ>, <text:NOUN>]
```

### Vocabulary

The role of the `Vocabulary` is convert word to vocabulary index.

```py
import chariot.transformer as ct


vocab = Vocabulary()

doc = [
    ["you", "are", "reading", "the", "book"],
    ["I", "don't", "know", "its", "title"],
]

vocab.fit(doc)
text = ["you", "know", "book", "title"]
indexed = vocab.transform(text)
inversed = vocab.inverse_transform(indexed)
```

```
> [4, 11, 8, 13]
> ['you', 'know', 'book', 'title']
```

You can specify the reserved word for unknown word etc and set parameters to limit vocabulary size. Example like following.

```py
vocab = Vocabulary(padding="_pad_", unknown="_unk_", min_df=1)
```

### Formatter

The role of the `Formatter` is adjust data for your model.

```py
import chariot.transformer as ct


formatter = ct.formatter.Padding(padding=0, length=5)

data = [
    [1, 2],
    [3, 4, 5],
    [1, 2, 3, 4, 5]
]

padded = formatter.transform(data)
```

```
> [[1 2 0 0 0]
   [3 4 5 0 0]
   [1 2 3 4 5]]
```

### Generator

The role of the `Generator` is generating the target / source data for your model.  
For example, when you train the language model, your target data is shifted source data.

```py
import chariot.transformer as ct


generator = ct.generator.ShiftedTarget(shift=1)
source, target = generator.generate([1, 2, 3, 4, 5], index=0, length=3)
```

```
> source
[1, 2, 3]
> target
[2, 3, 4]
```

Now you learned the role of each preprocessor. Then let's [make preprocessor pipeline by composing these](./make_preprocessor_pipeline.md).
