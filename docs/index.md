<h1 id="title">chariot</h1>
<p>
Deliver the ready-to-train data to your NLP model.
</p>
---

### Introduction

* Prepare Dataset
    * You can prepare typical NLP datasets through the [chazutsu](https://github.com/chakki-works/chazutsu).
* Build & Run Preprocess
    * You can build the preprocess pipeline like [scikit-learn Pipeline](http://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
    * Preprocesses for each dataset column are executed in parallel by [Joblib](https://pythonhosted.org/joblib/index.html).
    * Multi-language text tokenization is supported by [spaCy](https://spacy.io/).
* Format Batch
    * Sampling a batch from preprocessed dataset and format it to train the model (padding etc).
    * You can use pre-trained word vectors through the [chakin](https://github.com/chakki-works/chakin).

**chariot** enables you to concentrate on training your model!

![chariot flow](./images/chariot_feature.gif)

### License

[Apache License 2.0](https://github.com/chakki-works/chariot/blob/master/LICENSE)
