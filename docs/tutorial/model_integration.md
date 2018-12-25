<h2 id="title">Model Integration</h2>
---

You have to arrange the data format to feed it to your model. It's known as padding, one-hot vectorize etc.
Of course, `chariot` supports these feature.

### Make Data feeding process

`Feeder` enables it.

```py
import chariot.transformer as ct


preprocessed_data = {
    "label": [...],
    "review": [...]
}

feeder = Feeder({"label": ct.formatter.CategoricalLabel(num_class=5),
                 "review": ct.formatter.Padding.from_(padding=0, length=5)})
adjusted = feeder.transform(preprocessed_data)
```

`Feeder` supports batch iteration.

```py
for batch in feeder.iterate(preprocessed_data, batch_size=32, epoch=10):
    model.train_on_batch(batch["review"], batch["label"])

```

If you have `preprocessor`, you can use it to define each `formatter`s.

```py
import chariot.transformer as ct


feeder = Feeder({"label": ct.formatter.CategoricalLabel.from_(preprocessor),
                 "review": ct.formatter.Padding.from_(preprocessor, length=5)})
```

## Save and load a feeder

`Feeder` can bed saved to file like `preprocess`.

```py
feeder.save("/path/to/feeder.tar.gz")
feeder = Feeder.load("/path/to/feeder.tar.gz")
```

Congratulations!

You are ready to collaborate with `chariot`. Next is additional content to introduce [other convenient features of `chariot`](./prepare_resources.md).
