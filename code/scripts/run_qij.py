# Examine the value of q_ij for different perplexity


import os
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.spatial.distance import squareform

from icommon import compute_Q
from common.dataset import dataset
from common.dataset.constraint import gen_similar_links, gen_dissimilar_links
from MulticoreTSNE import MulticoreTSNE

from tensorboardX import SummaryWriter
from bokeh.plotting import figure, show, save
from bokeh.io import output_file

from collections import defaultdict


def examine_qij(dataset_name, writer=None, base_perp=None):
    _, X, _ = dataset.load_dataset(dataset_name)

    embedding_type = "normal" if base_perp is None else "chain"
    for perp in range(1, X.shape[0] // 3):
        file_name = f"{perp}.z" if base_perp is None else f"{base_perp}_to_{perp}.z"
        in_file = os.path.join(dir_path, embedding_type, dataset_name, file_name)
        data = joblib.load(in_file)
        Q = compute_Q(data["embedding"])
        nlogQ = -np.log(Q)
        nlogQ_normalized = nlogQ / np.max(nlogQ)

        if writer:
            print(f"Adding histogram for perp={perp} with {len(Q)} values")
            writer.add_histogram(f"qij", Q, global_step=perp)
            writer.add_histogram(f"qij_normalized", Q / Q.max(), global_step=perp)
            writer.add_histogram(f"-logqij", nlogQ, global_step=perp)
            writer.add_histogram(f"-logqij_normalized", nlogQ_normalized, perp)
        else:
            print("No tensorboardX debug info stored")


def extract_qij_for_some_pairs(
    dataset_name, normalized=False, use_log=False, base_perp=None, list_n_constraints=[]
):
    _, X, labels = dataset.load_dataset(dataset_name)
    perps = range(1, X.shape[0] // 3)

    # store all Q for different number of constraints
    Q_sim_all = defaultdict(list)
    Q_dis_all = defaultdict(list)

    embedding_type = "normal" if base_perp is None else "chain"

    # for each embedding, pick different pairs with different number of constraints
    for perp in perps:
        file_name = f"{perp}.z" if base_perp is None else f"{base_perp}_to_{perp}.z"
        in_file = os.path.join(dir_path, embedding_type, dataset_name, file_name)
        print(in_file)

        data = joblib.load(in_file)
        Q = compute_Q(data["embedding"])

        if use_log:
            Q = -np.log(Q)
        if normalized:
            Q /= Q.max()
        Q = squareform(Q)

        # store the q_ij for this `perp` for each num of constraint in Q_*_all[n_constraints]
        for n_constraints in list_n_constraints:
            sim = gen_similar_links(labels, n_constraints, include_link_type=False)
            dis = gen_dissimilar_links(labels, n_constraints, include_link_type=False)

            Q_sim = Q[sim[:, 0], sim[:, 1]]
            Q_dis = Q[dis[:, 0], dis[:, 1]]

            Q_sim_all[n_constraints].append(Q_sim)
            Q_dis_all[n_constraints].append(Q_dis)

        del Q

    # store all q_ij for both sim and dis links, together with list of all calculated perplexities
    out_name = (
        f"./q_ij/{dataset_name}"
        f"_{embedding_type}{base_perp if base_perp else ''}"
        f"{'_log' if use_log else ''}"
        f"{'_normalized' if normalized else ''}"
    )
    joblib.dump([perps, Q_sim_all, Q_dis_all], out_name)


def plot_qij_for_some_pairs(
    dataset_name, normalized=False, use_log=False, base_perp=None, list_n_constraints=[]
):
    embedding_type = "normal" if base_perp is None else "chain"
    in_name = (
        f"./q_ij/{dataset_name}"
        f"_{embedding_type}{base_perp if base_perp else ''}"
        f"{'_log' if use_log else ''}"
        f"{'_normalized' if normalized else ''}"
    )
    [list_perps, Q_sim_all, Q_dis_all] = joblib.load(in_name)
    print(list_perps, len(Q_sim_all), len(Q_dis_all))

    for n_constraints in list_n_constraints:
        qij_sim = np.array(Q_sim_all[n_constraints]).T
        qij_dis = np.array(Q_dis_all[n_constraints]).T
        print(qij_sim.shape, qij_dis.shape)

        p = figure(
            plot_width=900,
            plot_height=550,
            title=(
                f"{dataset_name}, tSNE-{embedding_type}{base_perp if base_perp else ''}"
                f" ({n_constraints} similar links, {n_constraints} dissimilar links)"
            ),
        )
        p.xaxis.axis_label = "Perplexity"
        p.yaxis.axis_label = (
            f"{'negative log q_ij' if use_log else 'q_ij'}{' normalized' if normalized else ''}"
        )

        p.multi_line(xs=[list(list_perps)] * len(qij_sim), ys=qij_sim.tolist(), line_alpha=0.5)
        p.multi_line(
            xs=[list(list_perps)] * len(qij_dis),
            ys=qij_dis.tolist(),
            line_alpha=0.5,
            line_color="red",
        )

        output_file(f"{in_name}_{2*n_constraints}constraints.html", title=dataset_name)
        save(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name")
    ap.add_argument("-bp", "--base_perp", default=None, type=int)
    ap.add_argument("-s", "--seed", default=2019, type=int)
    ap.add_argument("-r", "--run_id", default=9998, type=int)
    ap.add_argument("-lg", "--use_log", action="store_true")
    ap.add_argument("-nm", "--normalized", action="store_true")
    args = ap.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = f"{dir_path}/data"
    dataset.set_data_home(data_dir)

    # embedding_type = "normal" if args.base_perp is None else f"chain{args.base_perp}"
    # time_str = time.strftime("%b%d-%H:%M:%S", time.localtime())
    # log_dir = f"runs{args.run_id}/{args.dataset_name}/qij_{embedding_type}/{time_str}"
    # writer = SummaryWriter(log_dir=log_dir)

    # examine_qij(args.dataset_name, writer=writer, base_perp=args.perp)

    # remember to flush all data
    # writer.close()

    list_n_constraints = list(range(1, 11)) + [15, 20, 30, 50]
    kwargs = dict(
        dataset_name=args.dataset_name,
        use_log=args.use_log,
        normalized=args.normalized,
        base_perp=args.base_perp,
        list_n_constraints=list_n_constraints,
    )
    extract_qij_for_some_pairs(**kwargs)
    plot_qij_for_some_pairs(**kwargs)

