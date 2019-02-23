<h2 id="title">Make custom preprocessors</h2>
---

If prepared preprocessors are not enough to you, you can make custom preprocessors.

### Base classes for customizing

Following base classes are prepared to make custom preprocessor.

* TextPreprocessor: `TextNormalizer` and `TextFilter`
* Tokenizer: -
* TokenPreprocessor: `TokenNormalizer` and `TokenFilter`
* Vocabulary: -
* Formatter: `BaseFormatter`
* Generator: `SourceGenerator` and `TargetGenerator`

All you have to do is implements `apply` or `transform` method for each base class.
