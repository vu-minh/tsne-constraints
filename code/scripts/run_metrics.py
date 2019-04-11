"""Run metrics on the pre-calculated embeddings
"""
import os
import joblib
import numpy as np
import pandas as pd
from common.dataset import dataset
from common.metric.dr_metrics import DRMetric
from icommon import hyper_params


def calculate_metrics(
    dataset_name, metric_names, base_perp=None, earlystop="_earlystop"
):
    _, X, _ = dataset.load_dataset(dataset_name)
    N = X.shape[0]
    all_perps = range(1, N // 3)

    if base_perp is None:
        embedding_dir = f"{dir_path}/normal/{dataset_name}"
        in_name_prefix = ""
        out_name_sufix = ""
    else:
        embedding_dir = f"{dir_path}/chain/{dataset_name}"
        in_name_prefix = f"{base_perp}_to_"
        out_name_sufix = f"_base{base_perp}"

    result = []
    for perp in all_perps:
        in_name = f"{embedding_dir}/{in_name_prefix}{perp}{earlystop}.z"
        data = joblib.load(in_name)

        record = {
            "perplexity": perp,
            "kl_divergence": data["kl_divergence"],
            "bic": 2 * data["kl_divergence"] + np.log(N) * perp / N,
        }

        drMetric = DRMetric(X, data["embedding"])
        for metric_name in metric_names:
            metric_method = getattr(drMetric, metric_name)
            record[metric_name] = metric_method()

        result.append(record)

    df = pd.DataFrame(result).set_index("perplexity")
    df.to_csv(f"metrics/{dataset_name}{out_name_sufix}{earlystop}.csv")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-bp", "--base_perp", default=None, type=int)
    ap.add_argument("-e", "--earlystop", action="store_true")
    args = ap.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = f"{dir_path}/data"
    dataset.set_data_home(data_dir)

    dataset_name, base_perp = args.dataset_name, args.base_perp
    earlystop = "" if not args.earlystop else "_earlystop"
    metric_names = ["auc_rnx", "pearsonr", "mds_isotonic", "cca_stress", "sammon_nlm"]

    # calculate metrics for normal tSNE
    calculate_metrics(dataset_name, metric_names, base_perp=None, earlystop="")

    # calculate metrics for chain tSNE
    base_perps = (
        [base_perp]
        if base_perp is not None
        else hyper_params[dataset_name]["base_perps"]
    )
    for base_perp in base_perps:
        calculate_metrics(dataset_name, metric_names, base_perp, earlystop)
