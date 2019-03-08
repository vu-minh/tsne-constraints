"""Run multicore TSNE:
    https://github.com/DmitryUlyanov/Multicore-TSNE
"""
import os
import joblib
import numpy as np
from time import time
from MulticoreTSNE import MulticoreTSNE
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from common.dataset import dataset
import multiprocessing
from scipy.spatial.distance import pdist, squareform


DEV = False
USE_MULTICORE = True
fixed_seed = 2019
n_cpu_using = int(0.75 * multiprocessing.cpu_count())
MACHINE_EPSILON = np.finfo(np.double).eps

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{dir_path}/data"
dataset.set_data_home(data_dir)


def _compute_Q(X2d):
    """ Matrix Q in t-sne, used to calculate the prob. that a point `j`
    being neighbor of a point `i` (the value of Q[i,j])
    Make sure to call squareform(Q) before using it.
    """
    degrees_of_freedom = 1
    dist = pdist(X2d, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    return Q


def run_dataset(dataset_name):
    _, X, y = dataset.load_dataset(dataset_name)
    embedding_dir = f"{dir_path}/embeddings/{dataset_name}"
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)

    for perp in [30] if DEV else range(1, X.shape[0] // 2):
        start_time = time()
        if USE_MULTICORE:
            tsne = MulticoreTSNE(random_state=fixed_seed, perplexity=perp,
                                 n_jobs=n_cpu_using)
        else:
            tsne = TSNE(random_state=fixed_seed, perplexity=perp)
        tsne.fit_transform(X)
        running_time = time() - start_time
        print(f"{dataset_name}, {perp}, time: {running_time}s")

        result = dict(
            perplexity=perp,
            running_time=running_time,
            embedding=tsne.embedding_,
            Q=_compute_Q(tsne.embedding_),
            kl_divergence=tsne.kl_divergence_,
            n_jobs=tsne.n_jobs if USE_MULTICORE else 1,
            n_iter=tsne.n_iter_,
            learning_rate=tsne.learning_rate,
            random_state=tsne.random_state,
        )
        joblib.dump(value=result, filename=f"{embedding_dir}/{perp}.z")


def test_load_data(dataset_name, perp=30):
    _, _, y = dataset.load_dataset(dataset_name)
    embedding_dir = f"{dir_path}/embeddings/{dataset_name}"
    out_name = f"{embedding_dir}/{perp}.z"
    print("Test saved data from ", out_name)

    loaded = joblib.load(filename=out_name)
    for k, v in loaded.items():
        if k == 'embedding':
            Z = loaded[k]
            plt.scatter(Z[:, 0], Z[:, 1], c=y, alpha=0.3)
            plt.savefig(f"test_{'multicore' if USE_MULTICORE else 'sk'}.png")
        elif k == 'Q':
            Q = squareform(v)
            print('Q matrix: ', Q.shape, Q.min(), Q.max())
        else:
            print(k, v)


if __name__ == '__main__':
    dataset_name = 'FASHION200'
    run_dataset(dataset_name)
    if DEV:
        test_load_data(dataset_name, perp=30)
