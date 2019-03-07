"""Data selection and filtering, process the logics,
    find best perp according to user constraints.
"""

import os
import joblib
import numpy as np


dir_path = os.path.dirname(os.path.realpath(__file__))
embedding_dir = f"{dir_path}/embeddings"
DEV = False


def get_embedding(dataset_name, perp):
    in_name = f"{embedding_dir}/{dataset_name}/{int(perp)}.z"
    data = joblib.load(in_name)
    # TODO fix latter: not use DIGITS or rebuild embeddings for DIGITS
    Z = data if dataset_name == 'DIGITS' else data['embedding']
    return Z[:200] if DEV else Z
