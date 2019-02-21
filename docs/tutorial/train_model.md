<h2 id="title">Train your model</h2>
---

You can easily feed the preprocessed data to your model by `chariot`.

```py
for batch in dp(train_data.preprocess().iterate(batch_size=32, epoch=10):
    model.train_on_batch(batch["review"], batch["polarity"])
```

If you want to preprocess/format all the data before training, you can do it like following.

```py
formatted = dp(train_data).preprocess().format().processed

model.fit(formatted["review"], formatted["polarity"], batch_size=32,
          validation_split=0.2, epochs=15, verbose=2)
```

Congratulations!

You are ready to collaborate with `chariot`. Next is additional content to introduce [other convenient features of `chariot`](./prepare_resources.md).
