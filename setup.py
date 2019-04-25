from setuptools import setup


readme = ""
with open("README.md", encoding="utf-8") as f:
    readme = f.read()

setup(
    name="chariot",
    version="0.5.2",
    description="Deliver the ready-to-train data to your NLP model.",
    long_description=readme,
    long_description_content_type="text/markdown",
    keywords=["machine learning", "nlp", "natural language processing"],
    author="icoxfog417",
    author_email="icoxfog417@yahoo.co.jp",
    license="Apache License 2.0",
    packages=[
        "chariot",
        "chariot.resource",
        "chariot.transformer",
        "chariot.transformer.formatter",
        "chariot.transformer.generator",
        "chariot.transformer.text",
        "chariot.transformer.token",
        "chariot.transformer.tokenizer"
        ],
    url="https://github.com/chakki-works/chariot",
    install_requires=[
        "numpy>=1.14.4",
        "scipy>=1.1.0",
        "scikit-learn>=0.19.1",
        "spacy==1.9.0",
        "pandas>=0.23.1",
        "chazutsu>=0.8.2",
        "chakin>=0.0.6",
        "tqdm>=4.23.4"
    ],
)
