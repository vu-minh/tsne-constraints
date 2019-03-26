import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from common.dataset import dataset
from icommon import hyper_params


dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{dir_path}/data"
dataset.set_data_home(data_dir)


def _simple_scatter(Z, out_name, figsize=(6, 6), point_sizes=None, labels=None):
    plt.figure(figsize=figsize)
    plt.scatter(Z[:, 0], Z[:, 1], c=labels, s=point_sizes, alpha=0.25, cmap="jet")
    plt.tight_layout()
    plt.savefig(out_name, bbox_inches="tight", pad_inches=0)
    plt.close()


def _simple_loss(loss, out_name, figsize=(6, 1)):
    plt.figure(figsize=figsize)
    plt.plot(loss)
    plt.tight_layout()
    plt.savefig(out_name, bbox_inches="tight", pad_inches=0)
    plt.close()


def _scatter_with_loss(
    Z,
    loss,
    out_name,
    figsize=(6, 8),
    point_sizes=None,
    labels=None,
    scatter_title="",
    loss_title="",
):
    fig = plt.figure(figsize=figsize)
    grid = plt.GridSpec(4, 3, hspace=0.3)
    ax0 = fig.add_subplot(grid[:3, :])
    ax0.scatter(Z[:, 0], Z[:, 1], c=labels, s=point_sizes, alpha=0.25, cmap="jet")
    ax0.set_title(scatter_title)
    ax0.xaxis.tick_top()

    ax1 = fig.add_subplot(grid[3:, :])
    color1 = "tab:red"
    ax1.plot(loss[:25], color=color1)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_title(loss_title)

    ax2 = ax1.twinx()
    color2 = "tab:blue"
    ax2.plot(loss[25:], color=color2)
    ax2.tick_params(axis="y", labelcolor=color2)

    plt.savefig(out_name, bbox_inches="tight", pad_inches=0)
    plt.close()


def plot_embeddings(run_range=None, base_perp=None, force_rewrite=False):
    _, X, y = dataset.load_dataset(dataset_name)

    for perp in range(1, X.shape[0] // 3) if run_range is None else run_range:
        for earlystop in ["", "_earlystop"]:
            if base_perp is None:
                embedding_dir = f"{dir_path}/normal/{dataset_name}"
                file_name = f"{embedding_dir}/{perp}{earlystop}"
            else:
                embedding_dir = f"{dir_path}/chain/{dataset_name}"
                file_name = f"{embedding_dir}/{base_perp}_to_{perp}{earlystop}"

            if os.path.exists(f"{file_name}_all.png") and not force_rewrite:
                continue

            print("Plotting ", file_name)
            data = joblib.load(f"{file_name}.z")

            try:
                error_per_point = data["error_per_point"]
                error_as_point_size = (
                    MinMaxScaler(feature_range=(25, 150))
                    .fit_transform(error_per_point.reshape(-1, 1))
                    .reshape(1, -1)
                )

                progress_errors = data["progress_errors"]
                progress_errors = progress_errors[np.where(progress_errors > 0)]

                _scatter_with_loss(
                    Z=data["embedding"],
                    loss=progress_errors,
                    out_name=f"{file_name}_all.png",
                    point_sizes=error_as_point_size,
                    labels=y,
                    loss_title=(
                        f"final_loss={data['kl_divergence']:.3f},"
                        + f"  n_iter={data['n_iter']+1}"
                    ),
                )
            except KeyError:  # Exception:
                print("`error_per_point` or `progress_errors` are not available.")


def plot_extracted_info_by_key(base_perp, key, title=""):
    if base_perp is None:
        return

    file_name = f"./plot_chain/{dataset_name}_base{base_perp}_{key}"
    df = pd.read_csv(f"{file_name}.csv", index_col="perplexity")

    _, ax = plt.subplots(1, 1, figsize=(16, 4))
    df.plot(ax=ax)
    plt.title(title)
    plt.legend(ncol=4)
    plt.grid(True)
    plt.savefig(f"{file_name}.png")  # , bbox_inches="tight", pad_inches=0)


def plot_compare_kl_to_base(base_perp, key="kl_Qbase_Q", title="KL[Qbase || Q]"):
    if base_perp is None:
        return

    file_name = f"./plot_chain/{dataset_name}_base{base_perp}_{key}"
    df = pd.read_csv(f"{file_name}.csv", index_col="perplexity")

    _, ax = plt.subplots(1, 1, figsize=(16, 4))
    df.plot(ax=ax)
    plt.yscale("log")
    plt.title(title)
    plt.grid(True)
    plt.savefig(f"{file_name}.png")  # , bbox_inches="tight", pad_inches=0)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name")
    ap.add_argument("-x", "--dev", action="store_true")
    ap.add_argument("-p", "--test_perp")
    args = vars(ap.parse_args())

    dataset_name = args.get("dataset_name", "FASHION200")
    test_perp = args.get("test_perp", 30)
    DEV = args.get("dev", False)

    run_range = [test_perp] if DEV else None

    for base_perp in [None] + (
        [40] if DEV else hyper_params[dataset_name]["base_perps"]
    ):  # base_perp = None to plot the embedding `normal`
        plot_embeddings(run_range=run_range, base_perp=base_perp, force_rewrite=False)

        plot_extracted_info_by_key(
            base_perp, key="running_time", title="Compare running time"
        )
        plot_extracted_info_by_key(
            base_perp, key="n_iter", title="Compare number of running iterations"
        )

        plot_compare_kl_to_base(base_perp)
