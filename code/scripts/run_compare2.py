# compare the embeddings of different runs

import os
import joblib
import shutil
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist  # , squareform
from common.dataset import dataset
from icommon import hyper_params


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


embeddings_type = [
    ("normal", ""),
    ("normal", "_earlystop"),
    ("chain", ""),
    ("chain", "_earlystop"),
]


def _extract_for_one_perp(base_perp, perp, key):
    def _extract_by_key(embedding_type, earlystop):
        file_name = _get_filename_template(embedding_type).format(
            perp=perp, base_perp=base_perp, earlystop=earlystop
        )
        return joblib.load(file_name).get(key, None)

    result = {f"{k0}{k1}": _extract_by_key(k0, k1) for k0, k1 in embeddings_type}
    result["perplexity"] = perp
    if perp == base_perp:
        result["chain_earlystop"] = 0.0
    return result


def extract_from_embeddings(base_perp, run_range, keys=["running_time", "n_iter"]):
    # not efficient since load `normal` for each base_perp, but let it be
    for key in keys:
        result = [_extract_for_one_perp(base_perp, perp, key) for perp in run_range]
        df = pd.DataFrame(result).set_index("perplexity")
        df.to_csv(f"plot_chain/{dataset_name}_base{base_perp}_{key}.csv")


def extract_from_csv(base_perp, run_range, key="kl_Qbase_Q"):
    def _get_df_by_key(embedding_type, earlystop):
        file_name = f"plot_chain/{dataset_name}_{embedding_type}_{base_perp}{earlystop}"
        df = pd.read_csv(f"{file_name}.csv", index_col="perplexity")
        return df[[key]]

    df = pd.DataFrame(index=run_range)
    df.index.name = "perplexity"
    for embedding_type, earlystop in embeddings_type:
        dfx = _get_df_by_key(embedding_type, earlystop)
        dfx = dfx.rename(columns={key: f"{embedding_type}{earlystop}"})
        df = df.join(dfx, how="left")
    df.to_csv(f"plot_chain/{dataset_name}_base{base_perp}_{key}.csv")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name")
    ap.add_argument("-x", "--dev", action="store_true")
    ap.add_argument("-p", "--test_perp")
    args = vars(ap.parse_args())

    dataset_name = args.get("dataset_name", "FASHION200")
    test_perp = args.get("test_perp", 30)
    DEV = args.get("dev", False)

    _, X, _ = dataset.load_dataset(dataset_name)
    run_range = (
        [test_perp - 1, test_perp, test_perp + 1] if DEV else range(1, X.shape[0] // 3)
    )

    # compare KL, make sure to copy the needed files
    for base_perp in hyper_params[dataset_name]["base_perps"]:
        for earlystop in ["", "_earlystop"]:
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

                compare_kl_with_a_base(base_perp, run_range, embedding_type, earlystop)

    # create chart to compare running_time, n_iter, KL
    for base_perp in hyper_params[dataset_name]["base_perps"]:
        # extract running_time in 4cases: normal,normal_earlystop, chain,chain_earlystop
        extract_from_embeddings(base_perp, run_range, keys=["running_time", "n_iter"])

        # extract the comparation KL[Qbase||Q] for these 4 cases
        extract_from_csv(base_perp, run_range, key="kl_Qbase_Q")
