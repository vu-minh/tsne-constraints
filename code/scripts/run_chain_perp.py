# run tsne in chain

import os
import shutil  # for copying file
import joblib
import numpy as np
import pandas as pd
from time import time
from MulticoreTSNE import MulticoreTSNE
from matplotlib import pyplot as plt

# from matplotlib import ticker
from common.dataset import dataset
import multiprocessing
from scipy.spatial.distance import pdist  # , squareform
from common.metric.dr_metrics import DRMetric


# embeddings when run tSNE normal: in `normal` dir
# embeddings when run tSNE in chain: in `chain` dir

fixed_seed = 2019
n_cpu_using = int(0.75 * multiprocessing.cpu_count())

dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{dir_path}/data"
dataset.set_data_home(data_dir)

MACHINE_EPSILON = np.finfo(np.double).eps


def klpq(P, Q):
    return 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))


def compute_Q(X2d):
    """ Matrix Q in t-sne, used to calculate the prob. that a point `j`
    being neighbor of a point `i` (the value of Q[i,j])
    Make sure to call squareform(Q) before using it.
    """
    degrees_of_freedom = 1
    dist = pdist(X2d, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.0
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)
    return Q


def run_dataset(dataset_name, base_perp=30, run_range=range(31, 101)):
    _, X, y = dataset.load_dataset(dataset_name)
    chain_dir = f"{dir_path}/chain/{dataset_name}"
    if not os.path.exists(chain_dir):
        os.mkdir(chain_dir)

    # load init from perp=base_perp
    base_data = joblib.load(f"{dir_path}/normal/{dataset_name}/{base_perp}.z")
    Z_base = base_data["embedding"]

    for perp in run_range:
        if perp == base_perp:
            continue

        start_time = time()
        tsne = MulticoreTSNE(
            random_state=fixed_seed, perplexity=perp, n_jobs=n_cpu_using, init=Z_base
        )
        tsne.fit_transform(X)
        running_time = time() - start_time
        print(f"{dataset_name}, {perp}, time: {running_time}s")

        result = dict(
            perplexity=perp,
            running_time=running_time,
            embedding=tsne.embedding_,
            # Q=compute_Q(tsne.embedding_),
            kl_divergence=tsne.kl_divergence_,
            n_jobs=tsne.n_jobs,
            n_iter=tsne.n_iter_,
            learning_rate=tsne.learning_rate,
            random_state=tsne.random_state,
        )
        joblib.dump(value=result, filename=f"{chain_dir}/{base_perp}_to_{perp}.z")


def _scatter(Z, name, x_min=None, x_max=None, y_min=None, y_max=None, y=None):
    auto_scale = None in [x_min, x_max, y_min, y_max]
    plt.figure(figsize=(8, 8))
    plt.scatter(Z[:, 0], Z[:, 1], c=y, alpha=0.7, cmap="jet")
    if auto_scale:
        plt.savefig(f"{name}_autoscale.png")
    else:
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.savefig(f"{name}.png")
    plt.close()


def compare_kl_normal_chain(
    dataset_name, base_perp, run_range=range(1, 100), plot=True
):
    _, X, y = dataset.load_dataset(dataset_name)
    chain_dir = f"{dir_path}/chain/{dataset_name}"
    normal_dir = f"{dir_path}/embeddings/{dataset_name}"

    # get min, max of base embedding corresponding to base_perp
    base_data = joblib.load(f"{normal_dir}/{base_perp}.z")
    Z_base = base_data["embedding"]
    x_min, x_max = Z_base[:, 0].min(), Z_base[:, 0].max()
    y_min, y_max = Z_base[:, 1].min(), Z_base[:, 1].max()
    fig_limit = (x_min, x_max, y_min, y_max)

    result = []
    for perp in run_range:
        normal_file = f"{normal_dir}/{perp}.z"
        chain_file = f"{chain_dir}/{base_perp}_to_{perp}.z"
        if not os.path.exists(normal_file) or not os.path.exists(chain_file):
            print(
                "One of following files not found: \n",
                "\n".join([normal_file, chain_file]),
            )
            continue

        normal_data = joblib.load(normal_file)
        chain_data = joblib.load(chain_file)
        Z_normal = normal_data["embedding"]
        Z_chain = chain_data["embedding"]

        Q_normal = normal_data.get("Q", compute_Q(Z_normal))
        Q_chain = chain_data.get("Q", compute_Q(Z_chain))

        aucrnx_normal = DRMetric(X, Z_normal).auc_rnx()
        aucrnx_chain = DRMetric(X, Z_chain).auc_rnx()

        kl_normal_chain = klpq(Q_normal, Q_chain)
        kl_chain_normal = klpq(Q_chain, Q_normal)
        kl_sum = 0.5 * (kl_normal_chain + kl_chain_normal)

        record = {
            "perplexity": perp,
            "kl_normal_chain": kl_normal_chain,
            "kl_chain_normal": kl_chain_normal,
            "kl_sum": kl_sum,
            "aucrnx_normal": aucrnx_normal,
            "aucrnx_chain": aucrnx_chain,
            "running_time_normal": normal_data["running_time"],
            "running_time_chain": chain_data["running_time"],
        }
        result.append(record)
        print(
            "perp",
            perp,
            "diff_aucrnx",
            abs(aucrnx_normal - aucrnx_chain),
            "\tkl",
            kl_sum,
        )

        if plot:
            fig_name = f"{base_perp}_to_{perp}"
            _scatter(Z_normal, f"{normal_dir}/{fig_name}", y=y)
            _scatter(Z_normal, f"{normal_dir}/{fig_name}", *fig_limit, y=y)

            _scatter(Z_chain, f"{chain_dir}/{fig_name}", y=y)
            _scatter(Z_chain, f"{chain_dir}/{fig_name}", *fig_limit, y=y)
    df = pd.DataFrame(result).set_index("perplexity")
    df.to_csv(f"plot_chain/{dataset_name}_{base_perp}.csv")


def compare_kl_with_a_base(dataset_name, base_perp, embedding_type, run_range):
    if embedding_type not in ["normal", "chain"]:
        raise ValueError(f"{embedding_type} should in ['normal', 'chain']")

    # load base embedding
    base_data = joblib.load(f"{dir_path}/normal/{dataset_name}/{base_perp}.z")
    Q_base = base_data.get("Q", compute_Q(base_data["embedding"]))

    result = []
    embedding_dir = f"{dir_path}/{embedding_type}/{dataset_name}"
    for perp in run_range:
        in_name = f"{perp}" if embedding_type == "normal" else f"{base_perp}_to_{perp}"
        data = joblib.load(f"{embedding_dir}/{in_name}.z")
        Q = data.get("Q", compute_Q(data["embedding"]))

        kl_Q_Qbase = klpq(Q, Q_base)
        kl_Qbase_Q = klpq(Q_base, Q)
        kl_sum = 0.5 * (kl_Q_Qbase + kl_Qbase_Q)
        if perp == base_perp:
            print("KL: ", kl_Qbase_Q, kl_Q_Qbase)
        record = {
            "perplexity": perp,
            "kl_Q_Qbase": kl_Q_Qbase,
            "kl_Qbase_Q": kl_Qbase_Q,
            "kl_sum": kl_sum,
        }
        result.append(record)
    df = pd.DataFrame(result).set_index("perplexity")
    df.to_csv(f"plot_chain/{dataset_name}_{embedding_type}_{base_perp}.csv")


def viz_kl(dataset_name, base_perp, key_names, out_name):
    in_name_normal = f"plot_chain/{dataset_name}_normal_{base_perp}.csv"
    in_name_chain = f"plot_chain/{dataset_name}_chain_{base_perp}.csv"
    df_normal = pd.read_csv(in_name_normal, index_col="perplexity")
    df_chain = pd.read_csv(in_name_chain, index_col="perplexity")

    df = pd.merge(
        df_normal,
        df_chain,
        on="perplexity",
        how="inner",
        suffixes=("_normal", "_chain"),
    )
    df = df[df.index > 3]
    ind = df.index
    w = 0.3
    n_keys = len(key_names)
    _, axes = plt.subplots(n_keys, 1, figsize=(14, 3 * n_keys))

    for key_id, key_name in enumerate(key_names):
        ax = axes[key_id]
        p1 = ax.bar(ind, df[f"{key_name}_normal"], width=w)
        p2 = ax.bar(ind + w, df[f"{key_name}_chain"], width=w)
        ax.legend((p1[0], p2[0]), ("tSNE Normal", "tSNE chain"))

        ax.set_xticks(ind + w / 2)
        ax.autoscale_view()
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))

        ax.set_title(key_name)
        ax.set_xlabel("Perplexity")
        ax.set_ylabel(key_name)

    plt.tight_layout()
    plt.savefig(out_name)


def viz_kl_compare_chain_normal(dataset_name, base_perp, key_names, out_name):
    in_name = f"plot_chain/{dataset_name}_{base_perp}.csv"
    df = pd.read_csv(in_name, index_col="perplexity")
    df = df[df.index > 3]

    n_keys = len(key_names)
    w = 0.3
    ind = df.index
    _, axes = plt.subplots(n_keys, 1, figsize=(14, 3 * n_keys))
    for key_id, key_names in enumerate(key_names):
        ax = axes[key_id]
        ax.set_title(" v.s. ".join(key_names))
        p1 = ax.bar(ind, df[key_names[0]], width=w)
        if key_names:
            p2 = ax.bar(ind + w, df[key_names[1]], width=w)
            ax.legend((p1[0], p2[0]), ("tSNE Normal", "tSNE chain"))

        ax.set_xticks(ind + w / 2 if key_names[0] else 0)
        ax.autoscale_view()
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.set_xlabel("Perplexity")

    plt.tight_layout()
    plt.savefig(out_name)


if __name__ == "__main__":
    dataset_name = "DIGITS"
    chain_path = f"{dir_path}/chain/{dataset_name}"
    if not os.path.exists(chain_path):
        os.mkdir(chain_path)

    _, X, y = dataset.load_dataset(dataset_name)
    N = X.shape[0]

    min_perp, max_perp = 1, N // 3
    full_range = range(min_perp, max_perp)
    test_range = [29, 30, 31]

    for base_perp in [10, 20, 25, 30, 40, 50, 75, 100, 200]:
        # check if file {base_perp}_to_{base_perp}.z exists in chain dir
        target_file = f"{dir_path}/chain/{dataset_name}/{base_perp}_to_{base_perp}.z"
        if not os.path.exists(target_file):
            src_file = f"{dir_path}/normal/{dataset_name}/{base_perp}.z"
            if not os.path.exists(src_file):
                raise ValueError(f"{src_file} does not exist")
            shutil.copy2(src_file, target_file)
            print(f"Copy {src_file} to\n{target_file}")
        else:
            print(f"File {target_file} existed")

        run_dataset(dataset_name, base_perp, run_range=full_range)
        # backward_range = range(min_perp, base_perp)
        # forward_range = range(base_perp+1, max_perp)
        # run_dataset(dataset_name, base_perp, run_range=backward_range)
        # run_dataset(dataset_name, base_perp, run_range=forward_range)

        compare_kl_normal_chain(dataset_name, base_perp, run_range=full_range)

        compare_kl_with_a_base(dataset_name, base_perp, "normal", full_range)
        compare_kl_with_a_base(dataset_name, base_perp, "chain", full_range)

        key_names = ["kl_Qbase_Q", "kl_Q_Qbase", "kl_sum"]
        out_name = f"plot_chain/{dataset_name}_{base_perp}-kl_base_chain.png"
        viz_kl(dataset_name, base_perp, key_names, out_name)

        key_names = [
            ("running_time_normal", "running_time_chain"),
            ("kl_normal_chain", "kl_chain_normal"),
            ("aucrnx_normal", "aucrnx_chain"),
        ]
        out_name = f"plot_chain/{dataset_name}_{base_perp}-kl_chain_normal.png"
        viz_kl_compare_chain_normal(dataset_name, base_perp, key_names, out_name)
