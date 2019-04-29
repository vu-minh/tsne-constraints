# Examine the value of q_ij for different perplexity


import os
import time
import joblib
import argparse
import numpy as np

from icommon import compute_Q
from common.dataset import dataset
from MulticoreTSNE import MulticoreTSNE

from tensorboardX import SummaryWriter


def examine_qij(dataset_name, n_cpu_using=2, writer=None):
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
    writer = SummaryWriter(log_dir=log_dir)

    examine_qij(args.dataset_name, writer=writer)

    # remember to flush all data
    writer.close()
