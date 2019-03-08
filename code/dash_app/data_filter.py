"""Data selection and filtering, process the logics,
    find best perp according to user constraints.
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform


dir_path = os.path.dirname(os.path.realpath(__file__))
embedding_dir = f"{dir_path}/embeddings"
metric_dir = f"{dir_path}/metrics"
DEV = False


def get_list_embeddings(dataset_name):
    target_dir = os.path.join(embedding_dir, dataset_name)
    return [os.path.join(embedding_dir, dataset_name, f)
            for f in os.listdir(target_dir) if f.endswith('.z')]


def get_all_Q(list_embeddings):
    res = {}
    for in_name in list_embeddings:
        data = joblib.load(in_name)
        perp = data['perplexity']
        res[perp] = squareform(data['Q'])
    return res


def constraint_score(Q, sim, dis):
    s_sim = (0 if len(sim) == 0
             else np.sum(np.log(Q[sim[:, 0], sim[:, 1]])) / len(sim))
    s_dis = (0 if len(dis) == 0
             else - np.sum(np.log(Q[dis[:, 0], dis[:, 1]])) / len(dis))
    return s_sim, s_dis


def calculate_constraint_scores(dataset_name, sim_links, dis_links):
    list_embeddings = get_list_embeddings(dataset_name)
    all_Q = get_all_Q(list_embeddings)
    scores = []
    for perp, Q in all_Q.items():
        s_sim, s_dis = constraint_score(Q, sim_links, dis_links)
        scores.append({
            'perplexity': perp,
            'score_similar_links': s_sim,
            'score_dissimilar_links': s_dis,
            'score_all_links': 0.5 * (s_sim + s_dis)
        })
    return scores


def get_constraint_scores_df(dataset_name, constraints):
    sim_links = np.array([[int(link[0]), int(link[1])]
                          for link in constraints if link[2] == 'sim-link'])
    dis_links = np.array([[int(link[0]), int(link[1])]
                          for link in constraints if link[2] == 'dis-link'])
    scores = calculate_constraint_scores(dataset_name, sim_links, dis_links)
    df = pd.DataFrame(scores).set_index('perplexity')
    return df.sort_index()


def get_embedding(dataset_name, perp):
    in_name = f"{embedding_dir}/{dataset_name}/{int(perp)}.z"
    data = joblib.load(in_name)
    # TODO fix latter: not use DIGITS or rebuild embeddings for DIGITS
    Z = data if dataset_name == 'DIGITS' else data['embedding']
    return Z[:200] if DEV else Z


def get_metrics_df(dataset_name):
    # TODO cache this request
    df = pd.read_csv(f"{metric_dir}/{dataset_name}.csv",
                     index_col='perplexity')
    metric_names = [
        'kl_divergence',
        'auc_rnx',
        'pearsonr',
        'mds_isotonic',
        'cca_stress',
        # 'sammon_nlm',
        'bic'
    ]
    return df[metric_names].sort_index()
