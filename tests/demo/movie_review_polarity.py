import os
import sys
import shutil
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
import chazutsu
from chariot.storage import Storage


path = os.path.join(os.path.dirname(__file__), "../")
storage = Storage(path)
r = chazutsu.datasets.DUC2004().download(storage.data_path("raw"))
dataset = storage.chazutsu(r).train_dataset
print(dataset.to_dataframe().head(5))
shutil.rmtree(dataset.file_root)


