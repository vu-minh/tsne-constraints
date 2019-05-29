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
from MulticoreTSNE import MulticoreTSNE

from tensorboardX import SummaryWriter
from bokeh.plotting import figure, output_file, show

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


def test_plot_qij(dataset_name, normalized=False, use_log=False, n_constraints=20):
    _, X, _ = dataset.load_dataset(dataset_name)

    max_n_constraints = 50
    constraint_name = f"./links/auto_{dataset_name}_{max_n_constraints}sim_{max_n_constraints}dis.pkl"
    constraints = joblib.load(constraint_name)

    sim_links = np.array(
        [
            [int(link[0]), int(link[1])]
            for link in constraints
            if (link[2] == "sim-link" or link[2] == 1)
        ]
    )
    
    dis_links = np.array(
        [
            [int(link[0]), int(link[1])]
            for link in constraints
            if (link[2] == "dis-link" or link[2] == -1)
        ]
    )

    perps = range(1, X.shape[0] // 3)
    list_n_constraints = [2, 3, 5, 10, 15, 20, 30, 50]

    # store all Q for different number of constraints
    Q_sim_all = defaultdict(list)
    Q_dis_all = defaultdict(list)

    for perp in perps:
        in_file = os.path.join(dir_path, "normal", dataset_name, f"{perp}.z")
        data = joblib.load(in_file)
        Q = compute_Q(data["embedding"])

        if use_log:
            Q = -np.log(Q)
        if normalized:
            Q /= Q.max()
        Q = squareform(Q)

        # store the q_ij for this `perp` for each num of constraint in Q_*_all[n_constraints]
        for n_constraints in list_n_constraints:
            sim = sim_links[np.random.choice(max_n_constraints, n_constraints)]
            dis = dis_links[np.random.choice(max_n_constraints, n_constraints)]

            Q_sim = Q[sim[:, 0], sim[:, 1]]
            Q_dis = Q[dis[:, 0], dis[:, 1]]

            Q_sim_all[n_constraints].append(Q_sim)
            Q_dis_all[n_constraints].append(Q_dis)

    # plot the q_ij for each number of constraints
    for n_constraints in list_n_constraints:
        qij_sim = np.array(Q_sim_all[n_constraints]).T
        qij_dis = np.array(Q_dis_all[n_constraints]).T

        print(qij_sim.shape, qij_dis.shape)

        label = f"{'-log' if use_log else ''}q_ij{'_normalized' if normalized else ''}_{n_constraints}"
        output_file(f"./plots/{dataset_name}_{2*n_constraints}constraints_{label}.html", title=dataset_name)

        p = figure(
            plot_width=1200,
            plot_height=800,
            title=f"{dataset_name}, {2*n_constraints} constraints",
        )
        p.xaxis.axis_label = "Perplexity"
        p.yaxis.axis_label = label

        p.multi_line(xs=[list(perps)] * len(qij_sim), ys=qij_sim.tolist(), line_alpha=0.5)
        p.multi_line(
            xs=[list(perps)] * len(qij_dis),
            ys=qij_dis.tolist(),
            line_alpha=0.5,
            line_color="red",
        )
        show(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name")
    ap.add_argument("-x", "--dev", action="store_true")
    ap.add_argument("-p", "--perp", default=None, type=int)
    ap.add_argument("-t", "--embedding_type")
    ap.add_argument("-s", "--seed", default=2019, type=int)
    ap.add_argument("-r", "--run_id", default=9998, type=int)
    ap.add_argument("-n", "--n_constraints", default=5, type=int)
    args = ap.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = f"{dir_path}/data"
    dataset.set_data_home(data_dir)

    embedding_type = "normal" if args.perp is None else f"chain{args.perp}"
    time_str = time.strftime("%b%d-%H:%M:%S", time.localtime())
    # log_dir = f"runs{args.run_id}/{args.dataset_name}/qij_{embedding_type}/{time_str}"
    # writer = SummaryWriter(log_dir=log_dir)

    # examine_qij(args.dataset_name, writer=writer, base_perp=args.perp)

    # remember to flush all data
    # writer.close()

    test_plot_qij(
        args.dataset_name,
        use_log=True,
        normalized=True,
        n_constraints=args.n_constraints,
    )
