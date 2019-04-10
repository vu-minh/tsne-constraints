from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from time import time

import os
import joblib
import numpy as np
from scipy.spatial.distance import pdist, squareform

MACHINE_EPSILON = np.finfo(np.double).eps
print("MACHINE_EPSILON: ", MACHINE_EPSILON)


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
    return squareform(Q)


def test_tsne_Q_values():
    X, y = load_digits(return_X_y=True)
    # X = StandardScaler().fit_transform(X)
    X /= 255.0
    Z = TSNE(perplexity=500).fit_transform(X)
    Q = compute_Q(Z)
    print(f"[DEBUG]Q_min={np.min(Q[np.nonzero(Q)])}, Q_max={Q.max()}")

    # 😽 (14:10):scripts$ python run_tsne.py
    # MACHINE_EPSILON:  2.220446049250313e-16
    # Raw data:
    # [DEBUG]Q_min=4.053236149612854e-09, Q_max=7.134945480572868e-05
    # StandardScaler
    # [DEBUG]Q_min=5.3084445460460714e-09, Q_max=7.027090155865862e-05


if __name__ == "__main__":
    test_tsne_Q_values()
