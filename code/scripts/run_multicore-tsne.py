"""Run multicore TSNE:
    Folk from https://github.com/DmitryUlyanov/Multicore-TSNE
    Modified version with early-stop: https://github.com/vu-minh/Multicore-TSNE
"""
import os
import joblib
from time import time
import numpy as np
import multiprocessing

from MulticoreTSNE import MulticoreTSNE
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

from common.dataset import dataset


DEV = False
USE_MULTICORE = True
fixed_seed = 2019
n_cpu_using = int(0.75 * multiprocessing.cpu_count())

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{dir_path}/data"
dataset.set_data_home(data_dir)


def _simple_scatter(Z, out_name, figsize=(6, 6), point_sizes=None, labels=None):
    plt.figure(figsize=figsize)
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=point_sizes, alpha=0.25, cmap="jet")
    plt.tight_layout()
    plt.savefig(out_name)
    plt.close()


def run_dataset(dataset_name, plot=True):
    _, X, y = dataset.load_dataset(dataset_name)
    embedding_dir = f"{dir_path}/embeddings/{dataset_name}"
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)

    for perp in [test_perp] if DEV else range(1, X.shape[0] // 3):
        start_time = time()
        if USE_MULTICORE:
            tsne = MulticoreTSNE(
                verbose=1 if DEV else 0,
                random_state=fixed_seed,
                perplexity=perp,
                n_iter_without_progress=120,
                min_grad_norm=1e-04,
                n_jobs=n_cpu_using,
            )
        else:
            tsne = TSNE(random_state=fixed_seed, perplexity=perp)
        tsne.fit_transform(X)
        running_time = time() - start_time
        print(f"{dataset_name}, {perp}, time: {running_time}s")

        error_per_point = None
        error_as_point_size = None
        progress_errors = None
        try:
            error_per_point = tsne.error_per_point_
            error_as_point_size = (
                MinMaxScaler(feature_range=(25, 150))
                .fit_transform(error_per_point.reshape(-1, 1))
                .reshape(1, -1)
            )

            progress_errors = tsne.progress_errors_
            progress_errors = progress_errors[np.where(progress_errors > 0)]
        except AttributeError:
            print("`error_per_point_` or `progress_errors_` are not available.")

        out_name = f"{embedding_dir}/{perp}_earlystop"
        result = dict(
            perplexity=perp,
            running_time=running_time,
            embedding=tsne.embedding_,
            kl_divergence=tsne.kl_divergence_,
            n_jobs=tsne.n_jobs if USE_MULTICORE else 1,
            n_iter=tsne.n_iter_,
            learning_rate=tsne.learning_rate,
            random_state=tsne.random_state,
            progress_errors=progress_errors,
            error_per_point=error_per_point,
        )
        joblib.dump(value=result, filename=f"{out_name}.z")

        if plot:
            _simple_scatter(
                Z=tsne.embedding_,
                labels=y,
                point_sizes=error_as_point_size,
                out_name=f"{out_name}.png",
            )


def test_load_data(dataset_name, perp=30):
    _, _, y = dataset.load_dataset(dataset_name)
    embedding_dir = f"{dir_path}/embeddings/{dataset_name}"
    out_name = f"{embedding_dir}/{perp}_earlystop.z"
    print("\nTest loading saved data from ", out_name)

    loaded = joblib.load(filename=out_name)
    for k, v in loaded.items():
        if k not in ["embedding"]:
            print(k, v)

    error_per_point = loaded["error_per_point"]
    error_as_point_size = (
        None
        if error_per_point is None
        else (
            MinMaxScaler(feature_range=(10, 200))
            .fit_transform(error_per_point.reshape(-1, 1))
            .reshape(1, -1)
        )
    )

    _simple_scatter(
        Z=loaded["embedding"],
        labels=y,
        point_sizes=error_as_point_size,
        out_name=f"test_{'multicore' if USE_MULTICORE else 'sk'}.png",
    )


if __name__ == "__main__":
    if USE_MULTICORE:
        print("Runing MulticoreTSNE ", MulticoreTSNE.__version__)

    dataset_name = "DIGITS"
    test_perp = 40
    run_dataset(dataset_name, plot=True)
    if DEV:
        test_load_data(dataset_name, perp=test_perp)
