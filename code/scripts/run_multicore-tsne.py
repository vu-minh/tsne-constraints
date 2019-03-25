"""Run multicore TSNE:
    Folk from https://github.com/DmitryUlyanov/Multicore-TSNE
    Modified version with early-stop: https://github.com/vu-minh/Multicore-TSNE
"""
import os
import joblib
from time import time
import multiprocessing

from MulticoreTSNE import MulticoreTSNE
from sklearn.manifold import TSNE
from common.dataset import dataset


DEV = False
USE_MULTICORE = True
fixed_seed = 2019
n_cpu_using = int(0.75 * multiprocessing.cpu_count())

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{dir_path}/data"
dataset.set_data_home(data_dir)


def run_dataset(
    dataset_name, n_iter_without_progress=1000, min_grad_norm=1e-9, early_stop=False
):
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
                n_iter_without_progress=n_iter_without_progress,
                min_grad_norm=min_grad_norm,
                n_jobs=n_cpu_using,
            )
        else:
            tsne = TSNE(random_state=fixed_seed, perplexity=perp)
        tsne.fit_transform(X)
        running_time = time() - start_time
        print(f"{dataset_name}, {perp}, time: {running_time}s")

        error_per_point = None
        progress_errors = None
        try:
            error_per_point = tsne.error_per_point_
            progress_errors = tsne.progress_errors_
        except AttributeError:
            print("`error_per_point_` or `progress_errors_` are not available.")

        out_name = f"{embedding_dir}/{perp}{'_earlystop' if early_stop else ''}"
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


def test_load_data(dataset_name, perp=30):
    _, _, y = dataset.load_dataset(dataset_name)
    embedding_dir = f"{dir_path}/embeddings/{dataset_name}"
    out_name = f"{embedding_dir}/{perp}_earlystop.z"
    print("\nTest loading saved data from ", out_name)

    loaded = joblib.load(filename=out_name)
    print(loaded.keys())
    for k, v in loaded.items():
        if k not in ["embedding", "error_per_point", "progress_errors"]:
            print(k, v)


hyper_params = {
    "DIGITS": {"n_iter_without_progress": 150, "min_grad_norm": 1e-04},
    "FASHION200": {"n_iter_without_progress": 120, "min_grad_norm": 5e-04},
}

if __name__ == "__main__":
    if USE_MULTICORE:
        print("Runing MulticoreTSNE ", MulticoreTSNE.__version__)

    dataset_name = "FASHION200"
    test_perp = 40
    run_dataset(dataset_name, early_stop=False)
    run_dataset(dataset_name, early_stop=True, **hyper_params[dataset_name])

    if DEV:
        test_load_data(dataset_name, perp=test_perp)
