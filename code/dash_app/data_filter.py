"""Data selection and filtering, process the logics,
    find best perp according to user constraints.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


dir_path = os.path.dirname(os.path.realpath(__file__))
MACHINE_EPSILON = np.finfo(np.double).eps


def compute_Q(X2d):
    # TODO cache this function
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
    return squareform(Q)


def get_list_embeddings(dataset_name, base_perp=None, earlystop="_earlystop"):
    # in_name_prefix = f"{base_perp}_to_" if embedding_type == "chain" else ""
    # in_name = f"{embedding_dir}/{dataset_name}/{in_name_prefix}{perp}.z"
    embedding_type = "normal" if base_perp is None else "chain"
    embedding_dir = f"{dir_path}/{embedding_type}"
    target_dir = os.path.join(embedding_dir, dataset_name)
    return [
        os.path.join(embedding_dir, dataset_name, f)
        for f in os.listdir(target_dir)
        if f.endswith(f"{earlystop}.z")
        and (
            f.startswith(f"{base_perp}_to_")
            if base_perp is not None
            else not f.startswith(f"{base_perp}_to_")
        )
    ]


def constraint_score(Q, sim, dis):
    if len(Q.shape) < 2:
        Q = squareform(Q)

    s_sim = 0 if len(sim) == 0 else np.sum(np.log(Q[sim[:, 0], sim[:, 1]])) / len(sim)
    s_dis = 0 if len(dis) == 0 else -np.sum(np.log(Q[dis[:, 0], dis[:, 1]])) / len(dis)
    del Q
    return s_sim, s_dis


def calculate_constraint_scores(
    dataset_name, sim_links, dis_links, base_perp=None, earlystop="_earlystop"
):
    scores = []
    for in_name in get_list_embeddings(dataset_name, base_perp, earlystop):
        data = joblib.load(in_name)
        Q = data.get("Q", compute_Q(data["embedding"]))
        s_sim, s_dis = constraint_score(Q, sim_links, dis_links)
        scores.append(
            {
                "perplexity": data["perplexity"],
                "score_similar_links": s_sim,
                "score_dissimilar_links": s_dis,
                "score_all_links": 0.5 * (s_sim + s_dis),
            }
        )
    return scores


def get_constraint_scores_df(dataset_name, constraints, base_perp=None):
    sim_links = np.array(
        [[int(link[0]), int(link[1])] for link in constraints if link[2] == "sim-link"]
    )
    dis_links = np.array(
        [[int(link[0]), int(link[1])] for link in constraints if link[2] == "dis-link"]
    )
    scores = calculate_constraint_scores(dataset_name, sim_links, dis_links, base_perp)
    df = pd.DataFrame(scores).set_index("perplexity")
    return df.sort_index()


def get_embedding(dataset_name, perp, base_perp=None, earlystop="_earlystop"):
    if base_perp is None:
        embedding_dir = f"{dir_path}/normal"
        in_name_prefix = ""
    else:
        embedding_dir = f"{dir_path}/chain"
        in_name_prefix = f"{base_perp}_to_"
    in_name = f"{embedding_dir}/{dataset_name}/{in_name_prefix}{perp}{earlystop}.z"
    data = joblib.load(in_name)
    return data["embedding"]


def get_metrics_df(dataset_name, base_perp=None, earlystop="_earlystop"):
    # TODO cache this request
    in_name_prefix = "" if base_perp is None else f"_base{base_perp}"
    in_name = f"{dir_path}/metrics/{dataset_name}{in_name_prefix}{earlystop}.csv"
    df = pd.read_csv(in_name, index_col="perplexity")

    metric_names = [
        "kl_divergence",
        "auc_rnx",
        # "pearsonr",
        # "mds_isotonic",
        # "cca_stress",
        # 'sammon_nlm',
        "bic",
    ]
    return df[metric_names].sort_index()
