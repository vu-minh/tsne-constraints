# auto generate constraints

import os
import joblib
from common.dataset import dataset


def gen_constraints(dataset_name, n_gen=50):
    _, _, labels = dataset.load_dataset(args.dataset_name)
    pass


def test_show_constraints(dataset_name, n_gen):
    pass


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="")
    ap.add_argument("-n", "--n_links", default=0, type=int)
    ap.add_argument("-x", "--dev", action="store_true")
    args = ap.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    LINK_DIR = f"{dir_path}/links"
    DATA_DIR = f"{dir_path}/data"
    dataset.set_data_home(DATA_DIR)

    gen_constraints(args.dataset_name, n_gen=args.n_links)
    if args.dev:
        test_show_constraints(args.dataset_name, args.n_links)
