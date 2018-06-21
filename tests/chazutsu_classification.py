import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import chazutsu
from chariot.storage import Storage


ROOT_DIR = os.path.join(os.path.dirname(__file__), "./")
storage = Storage.setup_data_dir(ROOT_DIR)

r = chazutsu.datasets.MovieReview.polarity().download(storage.data_path("raw"))

y_train, x_train = r.train_data(split_target=True)
y_test, x_test = r.test_data(split_target=True)

vectorizer = TfidfVectorizer(stop_words="english")
x_train_v = vectorizer.fit_transform(x_train["review"].values)

classifier = LogisticRegression()
classifier.fit(x_train_v, y_train)

predict = classifier.predict(vectorizer.transform(x_test["review"].values))
score = metrics.accuracy_score(y_test, predict)

print(score)
