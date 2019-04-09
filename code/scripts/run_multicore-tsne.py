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
from icommon import hyper_params


def run_dataset(
    dataset_name, n_iter_without_progress=1000, min_grad_norm=1e-32, early_stop=""
):
    _, X, y = dataset.load_dataset(dataset_name)
    embedding_dir = f"{dir_path}/embeddings/{dataset_name}"
    if not os.path.exists(embedding_dir):
        os.mkdir(embedding_dir)

    base_min_grad_norm = min_grad_norm

    for perp in [test_perp] if DEV else range(1, X.shape[0] // 3):
        if early_stop != "":  # using early_stop
            adaptive_min_grad_norm = base_min_grad_norm * (10 ** (-(perp // 30)))
            print(
                f"perp={perp} ({perp//30}) adaptive_min_grad_norm={adaptive_min_grad_norm}"
            )
        else:
            adaptive_min_grad_norm = min_grad_norm

        start_time = time()
        if USE_MULTICORE:
            tsne = MulticoreTSNE(
                verbose=1 if DEV else 0,
                random_state=fixed_seed,
                perplexity=perp,
                n_iter_without_progress=n_iter_without_progress,
                min_grad_norm=adaptive_min_grad_norm,
                n_jobs=n_cpu_using,
                eval_interval=50,
                n_iter=1000,
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

        out_name = f"{dir_path}/tmp/{dataset_name}" if DEV else embedding_dir
        out_name += f"/{perp}{early_stop}"
        joblib.dump(value=result, filename=f"{out_name}.z")


def test_load_data(dataset_name, perp=30, early_stop=""):
    import numpy as np
    from run_plots import _simple_scatter, _simple_loss

    _, _, y = dataset.load_dataset(dataset_name)
    embedding_dir = f"{dir_path}/tmp/{dataset_name}"
    out_name = f"{embedding_dir}/{perp}{early_stop}"
    print("\nTest loading saved data from ", out_name)

    loaded = joblib.load(filename=f"{out_name}.z")
    _simple_scatter(Z=loaded["embedding"], out_name=f"{out_name}_scatter.png", labels=y)

    losses = loaded["progress_errors"]
    if losses is not None:
        losses = losses[np.where(losses > 0.0)]
        _simple_loss(loss=losses, out_name=f"{out_name}_loss.png", figsize=(6, 3))

    print(loaded.keys())
    for k, v in loaded.items():
        if k not in ["embedding", "error_per_point", "progress_errors"]:
            print(k, v)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-x", "--dev", action="store_true")
    ap.add_argument("-nm", "--non_multicore", action="store_true")
    ap.add_argument("-e", "--earlystop", action="store_true")
    ap.add_argument("-p", "--test_perp", default=30.0, type=float)
    args = ap.parse_args()

    dataset_name, USE_MULTICORE = args.dataset_name, not args.non_multicore
    test_perp, DEV = args.test_perp, args.dev

    fixed_seed = 2019
    n_cpu_using = int(0.75 * multiprocessing.cpu_count())

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = f"{dir_path}/data"
    dataset.set_data_home(data_dir)

    if USE_MULTICORE:
        print("Runing MulticoreTSNE ")  # , MulticoreTSNE.__version__)
    else:
        print("Running sklearn TSNE")

    if not args.earlystop:
        # run all (default 1000) iterations
        run_dataset(dataset_name, early_stop="")
    else:
        # run with early-stop condition in the config file (tcommon)
        run_dataset(
            dataset_name,
            early_stop="_earlystop",
            **hyper_params[dataset_name]["early_stop_conditions"],
        )

    if DEV:
        test_load_data(dataset_name, perp=test_perp, early_stop="_earlystop")
