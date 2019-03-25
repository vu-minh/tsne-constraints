# run tsne in chain

import os
import joblib
from time import time
import multiprocessing

from MulticoreTSNE import MulticoreTSNE
from common.dataset import dataset
from common import hyper_params


# embeddings when run tSNE normal: in `normal` dir
# embeddings when run tSNE in chain: in `chain` dir

fixed_seed = 2019
n_cpu_using = int(0.75 * multiprocessing.cpu_count())

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{dir_path}/data"
dataset.set_data_home(data_dir)


def run_dataset(
    dataset_name,
    base_perp=30,
    run_range=range(31, 101),
    n_iter_without_progress=1000,
    min_grad_norm=1e-9,
    early_stop=False,
):
    _, X, y = dataset.load_dataset(dataset_name)
    chain_dir = f"{dir_path}/chain/{dataset_name}"
    if not os.path.exists(chain_dir):
        os.mkdir(chain_dir)

    # load init from perp=base_perp
    user_early_stop = "_earlystop" if early_stop else ""
    in_name = f"{dir_path}/normal/{dataset_name}/{base_perp}{user_early_stop}.z"
    base_data = joblib.load(in_name)
    Z_base = base_data["embedding"]

    for perp in run_range:
        if perp == base_perp:
            continue

        start_time = time()
        tsne = MulticoreTSNE(
            random_state=fixed_seed,
            perplexity=perp,
            init=Z_base,
            n_iter_without_progress=n_iter_without_progress,
            min_grad_norm=min_grad_norm,
            n_jobs=n_cpu_using,
        )
        tsne.fit_transform(X)
        running_time = time() - start_time
        print(f"{dataset_name}, perp={perp}, {user_early_stop}, t={running_time:.3}s")

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
            n_jobs=tsne.n_jobs,
            n_iter=tsne.n_iter_,
            learning_rate=tsne.learning_rate,
            random_state=tsne.random_state,
            progress_errors=progress_errors,
            error_per_point=error_per_point,
        )
        out_name = f"{chain_dir}/{base_perp}_to_{perp}{user_early_stop}.z"
        joblib.dump(value=result, filename=out_name)


if __name__ == "__main__":
    DEV = False
    dataset_name = "FASHION200"
    chain_path = f"{dir_path}/chain/{dataset_name}"
    if not os.path.exists(chain_path):
        os.mkdir(chain_path)

    _, X, y = dataset.load_dataset(dataset_name)
    N = X.shape[0]

    run_range = [20] if DEV else range(1, N // 3)
    list_perps = [29, 30, 31] if DEV else hyper_params[dataset_name]["base_perps"]
    for base_perp in list_perps:
        run_dataset(dataset_name, base_perp, run_range=run_range, early_stop=False)
        run_dataset(
            dataset_name,
            base_perp,
            run_range=run_range,
            early_stop=True,
            **(hyper_params[dataset_name]["early_stop_conditions"]),
        )
