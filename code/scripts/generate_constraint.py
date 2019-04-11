# auto generate constraints

import os
import joblib
import random
import numpy as np
from common.dataset import dataset

# Reference: https://realpython.com/python-random/

SIM_LINK_TYPE = 1
DIS_LINK_TYPE = -1


def gen_simmilar_links(labels, n_links):
    min_class_id, max_class_id = labels.min(), labels.max()
    n_gen = 0
    links = []
    while n_gen < n_links:
        # pick random a class
        # random.randint(x, y) -> its range is [x, y]
        # random.randrange(x, y) -> its range is [x, y)
        c = random.randint(min_class_id, max_class_id)

        # filter the indices of points in this class
        (point_idx,) = np.where(labels == c)

        # pick random two indices in this class
        # do not use random.choices(items, k)
        # -> do with replacement -> duplicates are possible
        p1, p2 = random.sample(point_idx.tolist(), 2)

        # store the sampled indices with `link_type`=1
        links.append([p1, p2, SIM_LINK_TYPE])
        n_gen += 1

    return links


def gen_dissimilar_links(labels, n_links):
    min_class_id, max_class_id = labels.min(), labels.max()
    n_gen = 0
    links = []
    while n_gen < n_links:
        # pick 2 random different classes
        # not to use random.sample(items, k) to ensure random WITHOUT replacement
        c1, c2 = random.sample(range(int(min_class_id), int(max_class_id) + 1), 2)

        # filter point indices for each selected class, and take a sample for each class
        (idx1,) = np.where(labels == c1)
        (idx2,) = np.where(labels == c2)

        p1 = random.choices(idx1.tolist())[0]
        p2 = random.choices(idx2.tolist())[0]

        # store the generated link with `link_type`=-1
        links.append([p1, p2, DIS_LINK_TYPE])
        n_gen += 1

    return links


def gen_constraints(dataset_name, n_sim, n_dis):
    _, _, labels = dataset.load_dataset(args.dataset_name)

    sim_links = gen_simmilar_links(labels, n_sim)
    dis_links = gen_dissimilar_links(labels, n_dis)

    out_name = f"{LINK_DIR}/auto_{dataset_name}_{n_sim}sim_{n_dis}dis.pkl"
    joblib.dump(sim_links + dis_links, out_name)


def test_auto_generated_constraints(dataset_name, n_sim, n_dis):
    _, _, labels = dataset.load_dataset(dataset_name)

    in_name = f"{LINK_DIR}/auto_{dataset_name}_{n_sim}sim_{n_dis}dis.pkl"
    links = joblib.load(in_name)
    assert len(links) == n_sim + n_dis, (
        "Number of generated links is not correct."
        f"Expected {n_sim} + {n_dis} = {n_sim + n_dis}, got {len(links)}"
    )

    if len(links) > 0:
        assert len(links[0]) == 3, (
            "Expect 3 elements in a links [p1, p2, link_type],"
            f"but got {len(links[0])} for the first link"
        )

    for p0, p1, link_type in links:
        if link_type == SIM_LINK_TYPE:
            assert labels[p0] == labels[p1], "Labels of sim-link must be the same"
        elif link_type == DIS_LINK_TYPE:
            assert labels[p0] != labels[p1], "Labels of dis-link must not be the same"
        else:
            raise ValueError("`link_type` of auto-generated link must be +1 or -1")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-n", "--n_links", default=0, type=int, help="# of links each type")
    ap.add_argument("-ns", "--n_sim_links", default=None, type=int)
    ap.add_argument("-nd", "--n_dis_links", default=None, type=int)
    ap.add_argument("-s", "--seed", default=2019, type=int)
    ap.add_argument("-x", "--dev", action="store_true")
    args = ap.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    LINK_DIR = f"{dir_path}/links"
    DATA_DIR = f"{dir_path}/data"
    dataset.set_data_home(DATA_DIR)

    # always makesure to set random seed for reproducing
    random.seed(args.seed)

    n_sim = args.n_sim_links or args.n_links
    n_dis = args.n_dis_links or args.n_links
    gen_constraints(args.dataset_name, n_sim, n_dis)

    if args.dev:
        test_auto_generated_constraints(args.dataset_name, n_sim, n_dis)
