"""Data selection and filtering, process the logics,
    find best perp according to user constraints.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform


dir_path = os.path.dirname(os.path.realpath(__file__))
embedding_dir = f"{dir_path}/embeddings"
metric_dir = f"{dir_path}/metrics"
MACHINE_EPSILON = np.finfo(np.double).eps
DEV = False


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


def get_list_embeddings(dataset_name):
    target_dir = os.path.join(embedding_dir, dataset_name)
    return [
        os.path.join(embedding_dir, dataset_name, f)
        for f in os.listdir(target_dir)
        if f.endswith(".z")
    ]


def constraint_score(Q, sim, dis):
    if len(Q.shape) < 2:
        Q = squareform(Q)

    s_sim = 0 if len(sim) == 0 else np.sum(np.log(Q[sim[:, 0], sim[:, 1]])) / len(sim)
    s_dis = 0 if len(dis) == 0 else -np.sum(np.log(Q[dis[:, 0], dis[:, 1]])) / len(dis)
    del Q
    return s_sim, s_dis


def calculate_constraint_scores(dataset_name, sim_links, dis_links):
    scores = []
    for in_name in get_list_embeddings(dataset_name):
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


def get_constraint_scores_df(dataset_name, constraints):
    sim_links = np.array(
        [[int(link[0]), int(link[1])] for link in constraints if link[2] == "sim-link"]
    )
    dis_links = np.array(
        [[int(link[0]), int(link[1])] for link in constraints if link[2] == "dis-link"]
    )
    scores = calculate_constraint_scores(dataset_name, sim_links, dis_links)
    df = pd.DataFrame(scores).set_index("perplexity")
    return df.sort_index()


def get_embedding(dataset_name, perp):
    in_name = f"{embedding_dir}/{dataset_name}/{int(perp)}.z"
    data = joblib.load(in_name)
    # TODO fix latter: not use DIGITS or rebuild embeddings for DIGITS
    Z = data if dataset_name == "DIGITS" else data["embedding"]
    return Z[:200] if DEV else Z


def get_metrics_df(dataset_name):
    # TODO cache this request
    df = pd.read_csv(f"{metric_dir}/{dataset_name}.csv", index_col="perplexity")
    metric_names = [
        "kl_divergence",
        "auc_rnx",
        "pearsonr",
        "mds_isotonic",
        "cca_stress",
        # 'sammon_nlm',
        "bic",
    ]
    return df[metric_names].sort_index()
