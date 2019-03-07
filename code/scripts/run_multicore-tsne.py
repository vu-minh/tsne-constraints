"""Run multicore TSNE:
    https://github.com/DmitryUlyanov/Multicore-TSNE
"""
import os
import joblib
from time import time
from MulticoreTSNE import MulticoreTSNE
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from common.dataset import dataset


USE_MULTICORE = True
fixed_seed = 2019
dir_path = os.path.dirname(os.path.realpath(__file__))

dataset_name = 'FASHION2000'
data_dir = f"{dir_path}/data"
embedding_dir = f"{dir_path}/embeddings/{dataset_name}"
if not os.path.exists(embedding_dir):
    os.mkdir(embedding_dir)


perp = 40
dataset.set_data_home(data_dir)
_, X, y = dataset.load_dataset(dataset_name)

start_time = time()
if USE_MULTICORE:
    tsne = MulticoreTSNE(n_jobs=4, random_state=fixed_seed, perplexity=perp)
else:
    tsne = TSNE(random_state=fixed_seed, perplexity=perp)
tsne.fit_transform(X)
running_time = time() - start_time
print(f"Running time: {running_time}")

result = dict(
    running_time=running_time,
    embedding=tsne.embedding_,
    kl_divergence=tsne.kl_divergence_,
    n_jobs=tsne.n_jobs if USE_MULTICORE else 0,
    n_iter=tsne.n_iter_,
    learning_rate=tsne.learning_rate,
    random_state=tsne.random_state,
)

for k, v in result.items():
    if k != 'embedding':
        print(k, v)

out_name = f"{embedding_dir}/{perp}.z"
joblib.dump(value=result, filename=out_name, compress=True)

# test load data
print("Test saved data from ", out_name)
loaded = joblib.load(filename=out_name)
for k, v in loaded.items():
    if k == 'embedding':
        Z = loaded[k]
        plt.scatter(Z[:, 0], Z[:, 1], c=y)
        plt.savefig(f"test_{'multicore' if USE_MULTICORE else 'sklearn'}.png")
    else:
        print(k, v)
