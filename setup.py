from setuptools import setup


setup(
    name="chariot",
    version="0.0.1",
    description="Speedy data processing tool for NLP tasks",
    keywords=["machine learning", "nlp", "natural language processing"],
    author="icoxfog417",
    author_email="icoxfog417@yahoo.co.jp",
    license="Apache License 2.0",
    packages=[
        "chariot",
        "chariot.corpus",
        "chariot.preprocessor",
        "chariot.storage",
        "chariot.tokenizer",
        ],
    url="https://github.com/chakki-works/chariot",
    install_requires=[
        "numpy>=1.14.4",
        "spacy>=2.0.11",
        "tqdm>=4.23.4"
    ],
)
