# Commonly used utils

import os
import numpy as np
from scipy.spatial.distance import pdist, squareform

# hypter params for running tSNE with different params (and in chain)
# for different datasets
hyper_params = {
    "COIL20": {
        # "early_stop_conditions": {
        #     "n_iter_without_progress": 200,
        #     "min_grad_norm": 1e-5,  # perp=200: 5e-5, perp=300: 1e-10
        #     # set min_grad_norm 1e-05 to test adaptive grad_norm
        # },
        "base_perps": [30, 10],  # [20, 50, 75],
        "selected_perps": {  # for plotting example embeddings
            10: [5, 15, 50, 150, 250, 450],
            20: [5, 15, 50, 120, 250, 450],
            30: [5, 20, 50, 150, 250, 450],
        },
    },
    "DIGITS": {
        # "early_stop_conditions": {
        #     "n_iter_without_progress": 150,
        #     "min_grad_norm": 1e-04,
        # },
        "base_perps": [30, 10]  # [20, 50, 75],
    },
    "FASHION200": {
        # "early_stop_conditions": {
        #     "n_iter_without_progress": 120,
        #     "min_grad_norm": 5e-04,
        # },
        "base_perps": [10, 30],  # [20,40],
        "selected_perps": {  # for plotting example embeddings
            10: [5, 15, 25, 35, 50, 65],
            20: [5, 15, 25, 35, 50, 65],
            30: [5, 10, 25, 35, 50, 65],
        },
    },
    "QUICKDRAW500": {
        # "early_stop_conditions": {
        #     "n_iter_without_progress": 120,
        #     "min_grad_norm": 5e-04,
        # },
        "base_perps": [10, 30],  # [20,40],
        "selected_perps": {  # for plotting example embeddings
            10: [5, 15, 25, 35, 50, 65],
            20: [5, 15, 25, 35, 50, 65],
            30: [5, 10, 25, 35, 50, 65],
        },
    },
    "BREAST_CANCER": {
        "early_stop_conditions": {
            "n_iter_without_progress": 120,
            "min_grad_norm": 2e-04,
        },
        "base_perps": [30, 10],  # [20, 50],
    },
    "MPI": {
        "early_stop_conditions": {
            "n_iter_without_progress": 50,
            "min_grad_norm": 5e-04,
        },
        "base_perps": [5, 10, 20, 30],
    },
    "DIABETES": {
        "early_stop_conditions": {
            "n_iter_without_progress": 100,
            "min_grad_norm": 2e-04,
        },
        "base_perps": [10, 20, 30, 40],
    },
    "COUNTRY2014": {
        # "early_stop_conditions": {
        #     "n_iter_without_progress": 50,
        #     "min_grad_norm": 2e-04,
        # },
        "base_perps": [5, 10, 20, 30]
    },
}


# utils functions for calculate Q_ij in tSNE

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


def get_embedding_filename_template(
    dir_path, embedding_type, dataset_name, base_perp, perp, earlystop
):
    return os.path.join(
        dir_path,
        embedding_type,
        dataset_name,
        ("{base_perp}_to_{perp}" if embedding_type == "chain" else "{perp}")
        + "{earlystop}.z",
    )
