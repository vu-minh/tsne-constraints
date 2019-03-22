# run tsne in chain

import os
import joblib
import numpy as np
from time import time
import multiprocessing

from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from MulticoreTSNE import MulticoreTSNE

from common.dataset import dataset


# embeddings when run tSNE normal: in `normal` dir
# embeddings when run tSNE in chain: in `chain` dir

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


def run_dataset(dataset_name, base_perp=30, run_range=range(31, 101), plot=True):
    _, X, y = dataset.load_dataset(dataset_name)
    chain_dir = f"{dir_path}/chain/{dataset_name}"
    if not os.path.exists(chain_dir):
        os.mkdir(chain_dir)

    # load init from perp=base_perp
    base_data = joblib.load(f"{dir_path}/normal/{dataset_name}/{base_perp}_earlystop.z")
    Z_base = base_data["embedding"]

    for perp in run_range:
        if perp == base_perp:
            continue

        start_time = time()
        tsne = MulticoreTSNE(
            random_state=fixed_seed,
            perplexity=perp,
            init=Z_base,
            n_iter_without_progress=120,
            min_grad_norm=1e-04,
            n_jobs=n_cpu_using,
        )
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

        out_name = f"{chain_dir}/{base_perp}_to_{perp}_earlystop"
        result = dict(
            perplexity=perp,
            running_time=running_time,
            embedding=tsne.embedding_,
            kl_divergence=tsne.kl_divergence_,
            n_jobs=tsne.n_jobs,
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
                out_name=f"{out_name}_autoscale.png",
            )


hyper_params = {"FASHION200": {"base_perps": [10, 20, 30, 40]}}


if __name__ == "__main__":
    dataset_name = "FASHION200"
    chain_path = f"{dir_path}/chain/{dataset_name}"
    if not os.path.exists(chain_path):
        os.mkdir(chain_path)

    _, X, y = dataset.load_dataset(dataset_name)
    N = X.shape[0]

    min_perp, max_perp = 1, N // 3
    full_range = range(min_perp, max_perp)
    test_range = [29, 30, 31]
    base_perps = hyper_params[dataset_name]["base_perps"]

    for base_perp in base_perps:
        run_dataset(dataset_name, base_perp, run_range=full_range, plot=False)
