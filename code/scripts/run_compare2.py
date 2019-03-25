# compare the embeddings of different runs

import os
import joblib
import shutil
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist  # , squareform
from common.dataset import dataset
from common import hyper_params


# embeddings when run tSNE normal: in `normal` dir
# embeddings when run tSNE in chain: in `chain` dir

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


def _get_filename_template(embedding_type):
    return os.path.join(
        dir_path,
        embedding_type,
        dataset_name,
        ("{base_perp}_to_{perp}" if embedding_type == "chain" else "{perp}")
        + "{earlystop}.z",
    )


def compare_kl_with_a_base(base_perp, run_range, embedding_type, earlystop=""):
    # load base embedding
    base_name = _get_filename_template("normal").format(
        perp=base_perp, earlystop=earlystop
    )
    base_data = joblib.load(base_name)
    Q_base = base_data.get("Q", compute_Q(base_data["embedding"]))
    print(f"Compare KL[Qbase||Q] with for `{embedding_type}` Qbase from {base_name}")

    result = []
    for perp in run_range:
        chain_name = _get_filename_template(embedding_type).format(
            base_perp=base_perp, perp=perp, earlystop=earlystop
        )
        data = joblib.load(chain_name)
        Q = data.get("Q", compute_Q(data["embedding"]))

        kl_Q_Qbase = klpq(Q, Q_base)
        kl_Qbase_Q = klpq(Q_base, Q)
        kl_sum = 0.5 * (kl_Q_Qbase + kl_Qbase_Q)
        record = {
            "perplexity": perp,
            "kl_Q_Qbase": kl_Q_Qbase,
            "kl_Qbase_Q": kl_Qbase_Q,
            "kl_sum": kl_sum,
        }
        result.append(record)
    df = pd.DataFrame(result).set_index("perplexity")
    df.to_csv(f"plot_chain/{dataset_name}_{embedding_type}_{base_perp}{earlystop}.csv")


if __name__ == "__main__":
    dataset_name = "FASHION200"
    chain_path = f"{dir_path}/chain/{dataset_name}"
    if not os.path.exists(chain_path):
        os.mkdir(chain_path)

    _, X, _ = dataset.load_dataset(dataset_name)
    N = X.shape[0]

    min_perp, max_perp = 1, N // 3
    full_range = range(min_perp, max_perp)
    test_range = [29, 30, 31]
    base_perps = hyper_params[dataset_name]["base_perps"]

    for earlystop in ["", "_earlystop"]:
        for base_perp in base_perps:
            target_file = _get_filename_template("chain").format(
                base_perp=base_perp, perp=base_perp, earlystop=earlystop
            )
            src_file = _get_filename_template("normal").format(
                perp=base_perp, earlystop=earlystop
            )
            for embedding_type in ["normal", "chain"]:
                if embedding_type == "chain":
                    try:
                        shutil.copy2(src_file, target_file)
                    except Exception:
                        print(f"Error copying file: {src_file} -> {target_file}")

                compare_kl_with_a_base(base_perp, full_range, embedding_type, earlystop)
