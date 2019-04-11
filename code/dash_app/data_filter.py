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


def get_embedding(dataset_name, perp, earlystop, base_perp=None, return_all=False):
    """Util function to get individual embedding file (.z) for the corresponding dataset

    :params: base_perp: base perplexity in case of chain-tSNE

    :returns: a dictionary of precalculated info if `return_all` is True,
              otherwise, only the 2D embedding in data["embedding"] is returned
    """
    if base_perp is None:
        embedding_dir = f"{dir_path}/normal"
        in_name_prefix = ""
    else:
        embedding_dir = f"{dir_path}/chain"
        in_name_prefix = f"{base_perp}_to_"
    in_name = f"{embedding_dir}/{dataset_name}/{in_name_prefix}{perp}{earlystop}.z"
    data = joblib.load(in_name)
    return data if return_all else data["embedding"]


def _compute_Q(X2d):
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


def _get_list_embeddings(dataset_name, earlystop, base_perp=None):
    """Loop over the directory for the precalculated embeddings
    and return the full path to the .z file.

    :params: base_perp: base perplexity in case of chain-tSNE in `chain` dir,
             otherwise, use normal embeddings in `normal` dir

    :returns: List[str] of full path to the embeddings, then being fed to joblib.load()
    """
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


def constraint_score(Q, sim, dis, debug=False):
    """Core function to calculate constraint scores.

    :params: Q np.array(float) matrix Q = [q_ij] of size [N, N], N: number of datapoints
    :params: sim List[int, int] list of similar links
    :params: dis List[int, int] list of dissimilar links
    :params: debug defaults to False, flag to return detailed scores for each link

    :returns: score for each type of link,
              with or without detailed score for each individual
    """
    # print(f"[DEBUG]Q_min={np.min(Q[np.nonzero(Q)])}, Q_max={Q.max()}")
    if len(Q.shape) < 2:
        Q = squareform(Q)

    def score(links):
        return 0 if len(links) == 0 else np.mean(np.log(Q[links[:, 0], links[:, 1]]))

    def score_debug(links, link_type=""):
        score_link = 0.0
        debug_info = []
        for link in links:
            p0, p1 = link
            q = Q[p0, p1]
            s = np.log(q)
            debug_info.append([str(p0), str(p1), link_type, f"{q:1.1E}", f"{s:2.2f}"])
            score_link += s
        if len(links) > 0:
            score_link /= len(links)
        print(link_type, debug_info, score_link)
        return score_link, debug_info

    if not debug:
        return score(sim), -score(dis)
    else:
        print(score(sim), -score(dis))
        s_sim, debug_sim = score_debug(sim, "sim-link")
        s_dis, debug_dis = score_debug(dis, "dis-link")
        return s_sim, -s_dis, debug_sim + debug_dis


def calculate_constraint_scores(
    dataset_name, sim_links, dis_links, earlystop, base_perp=None
):
    """Calculate constraint scores for all perplexities of a given dataset.

    :returns: list of dictionary, with fields being different types of constraint scores
    which can be wrapped into a DataFrame and stored in csv file.
    """
    scores = []
    for in_name in _get_list_embeddings(dataset_name, earlystop, base_perp):
        data = joblib.load(in_name)
        Q = data.get("Q", _compute_Q(data["embedding"]))
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


def get_constraint_scores_df(
    dataset_name, constraints, earlystop, base_perp=None, debug=False
):
    """Calculate constraint score for the input `constraints`

    :returns: DataFrame with 4 columns, sorted by index
        + perplexity (index)
        + score_similar_links
        + score_dissimilar_links
        + score_all_links
    """
    sim_links = np.array(
        [[int(link[0]), int(link[1])] for link in constraints if link[2] == "sim-link"]
    )
    dis_links = np.array(
        [[int(link[0]), int(link[1])] for link in constraints if link[2] == "dis-link"]
    )
    scores = calculate_constraint_scores(
        dataset_name, sim_links, dis_links, earlystop, base_perp
    )
    df = pd.DataFrame(scores).set_index("perplexity")

    debug_links = None
    if debug:
        best_perp = df["score_all_links"].idxmax()
        embedding = get_embedding(dataset_name, best_perp, earlystop, base_perp)
        Q = _compute_Q(embedding)
        _, _, debug_links = constraint_score(Q, sim_links, dis_links, debug=True)
    return df.sort_index(), debug_links


def get_metrics_df(dataset_name, earlystop, base_perp=None):
    """Read the precalculated metrics in the csv corresponding csv file.

    :returns: DataFrame with indexed column `perplexity` and the metric columns as below
    """
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
