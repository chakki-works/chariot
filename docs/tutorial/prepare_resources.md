<h2 id="title">Prepare Resources</h2>
---

`chariot` have the feature to prepare resources for NLP research. Specifically, data and pretrained vectors.

### Download NLP dataset

`chariot` can collaborate with [`chazutsu`](https://github.com/chakki-works/chazutsu) that is NLP datasets downloader.

```py
import chazutsu
from chariot.storage import Storage

storage = Storage.setup_data_dir(ROOT_DIR)
r = chazutsu.datasets.MovieReview.polarity().download(storage.path("raw"))
r.train_data().head(3)
```

```
	polarity	review
0	0	synopsis : an aging master art thief , his sup...
1	0	plot : a separated , glamorous , hollywood cou...
2	0	a friend invites you to a movie . this film wo...
```

### Download Pretrained Word Vector

`chariot` can load the pretrained word vector by collaborating with [`chakin`](https://github.com/chakki-works/chakin).

```py
storage = Storage("path/to/project/root/data")
vec_path = storage.chakin(name="GloVe.6B.200d")  # download word vector

vocab = Vocabulary.from_file("path/to/vocabulary")
embedding = vocab.make_embedding(storage.path("external/glove.6B.200d.txt"))
```
