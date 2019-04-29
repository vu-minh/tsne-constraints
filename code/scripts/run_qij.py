# Examine the value of q_ij for different perplexity


import os
import time
import joblib
import argparse
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from icommon import compute_Q
from common.dataset import dataset
from MulticoreTSNE import MulticoreTSNE

from tensorboardX import SummaryWriter


def examine_qij(dataset_name, writer=None):
    _, X, _ = dataset.load_dataset(dataset_name)

    for perp in range(1, X.shape[0] // 3):
        in_file = os.path.join(dir_path, "normal", dataset_name, f"{perp}.z")
        data = joblib.load(in_file)
        Q = compute_Q(data["embedding"])
        logQ = np.log(Q)

        if writer:
            print(f"Adding histogram for perp={perp} with {len(Q)} values")
            writer.add_histogram(f"qij", Q, global_step=perp)
            writer.add_histogram(f"qij_normalized", Q / Q.max(), global_step=perp)
            writer.add_histogram(f"logqij", logQ, global_step=perp)
            writer.add_histogram(f"logqij_normalized", logQ / logQ.max(), perp)
        else:
            print("No tensorboardX debug info stored")


def test_plot_qij(dataset_name, normalized=False, use_log=False):
    _, X, _ = dataset.load_dataset(dataset_name)

    plt.figure(figsize=(18, 18))

    perps = range(1, X.shape[0] // 3)
    qij = []

    for perp in perps:
        in_file = os.path.join(dir_path, "normal", dataset_name, f"{perp}.z")
        data = joblib.load(in_file)
        Q = compute_Q(data["embedding"])
        if use_log:
            Q = np.log(Q)
        if normalized:
            Q /= Q.max()
        qij.append(Q)

    qij = np.array(qij).T
    print(qij.shape)

    from bokeh.plotting import figure, output_file, show

    p = figure(plot_width=1200, plot_height=800, title=f"{dataset_name} ({qij.shape})")
    p.multi_line(xs=[list(perps)] * len(qij), ys=qij.tolist(), line_alpha=0.01)

    label = f"{'log ' if use_log else ''} q_ij {' normalized' if normalized else ''}"
    p.xaxis.axis_label = "Perplexity"
    p.yaxis.axis_label = label

    output_file(f"./plots/{dataset_name}_{label}.html", title=dataset_name)
    show(p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name")
    ap.add_argument("-x", "--dev", action="store_true")
    ap.add_argument("-p", "--perp", default=30, type=int)
    ap.add_argument("-s", "--seed", default=2019, type=int)
    ap.add_argument("-r", "--run_id", default=9998, type=int)
    args = ap.parse_args()

    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = f"{dir_path}/data"
    dataset.set_data_home(data_dir)

    time_str = time.strftime("%b%d/%H:%M:%S", time.localtime())
    log_dir = f"runs{args.run_id}/{args.dataset_name}/qij_{time_str}"
    # writer = SummaryWriter(log_dir=log_dir)

    # examine_qij(args.dataset_name, writer=writer)

    # remember to flush all data
    # writer.close()

    test_plot_qij(args.dataset_name, use_log=False, normalized=True)
