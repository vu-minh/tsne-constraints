import os
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from common.dataset import dataset
from icommon import hyper_params


OUTPUT_EXT = "png"  # "pdf" for paper
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


def plot_metamap(ax, all_perps=[], base_perp=None, earlystop=""):
    print(f"Generate meta-plot for {dataset_name} with params: ", locals())

    if base_perp is None:
        embedding_dir = f"{dir_path}/normal/{dataset_name}"
        in_name_prefix = ""
        out_name_sufix = ""
    else:
        embedding_dir = f"{dir_path}/chain/{dataset_name}"
        in_name_prefix = f"{base_perp}_to_"
        out_name_sufix = f"_base{base_perp}"

    # prepare all pre-calculated embeddings
    all_Z = []
    all_perps = []
    all_losses = []
    for perp in all_perps:
        in_name = f"{embedding_dir}/{in_name_prefix}{perp}{earlystop}.z"
        data = joblib.load(in_name)
        all_perps.append(perp)
        all_Z.append(data["embedding"].ravel())
        all_losses.append(data["kl_divergence"])

    # using all_Z as features for meta-tSNE
    # all_Z = StandardScaler().fit_transform(all_Z)
    meta_Z = TSNE(perplexity=10).fit_transform(all_Z)

    out_name = f"{dataset_name}{out_name_sufix}{earlystop}"
    # plt.figure(figsize=(6, 6))
    ax.scatter(meta_Z[:, 0], meta_Z[:, 1], c=all_perps)
    ax.set_title(out_name)
    cbar = plt.colorbar()
    cbar.ax.set_title("Perplexity")
    # plt.tight_layout()
    # plt.savefig(f"./plot_chain/metamap/{out_name}.png")


def plot_an_embedding(ax, perp, base_perp=None, name=""):
    pass


def plot_metamap_with_some_perps(
    perps=[], base_perp=30, with_metamap=True, out_name="metaplot"
):
    """Plot examples of chain-tsne with perp in `perps`.
    If `with_metamap` is set, the metamap of all chain-tsne will be shown.
    """
    (n_rows, n_cols) = (2, 5 if with_metamap else 3)
    plt.figure(figsize=(4 * n_cols, 5 * n_rows))
    if with_metamap:
        ax0 = plt.subplot2grid((n_rows, n_cols), (0, 3), rowspan=2, colspan=2)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
    for i in range(6):
        row_i, col_i = i // 3, i % 3
        ax1 = plt.subplot2grid((n_rows, n_cols), (row_i, col_i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.tight_layout()
    plt.savefig(f"{out_name}.{OUTPUT_EXT}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset_name", default="FASHION200")
    ap.add_argument("-x", "--dev", action="store_true")
    ap.add_argument("-e", "--earlystop", action="store_true")
    ap.add_argument("-p", "--test_perp", default=30, type=int)
    ap.add_argument("-bp", "--base_perp", default=None, type=int)
    args = ap.parse_args()

    dataset_name = args.dataset_name
    test_perp = args.test_perp
    base_perp = args.base_perp
    DEV = args.dev
    earlystop = "_earlystop" if args.earlystop else ""

    out_name = f"../../figures/chain_tsne_examples_metaplot"
    plot_metamap_with_some_perps(
        perps=[5, 10, 30, 50, 100, 200],
        base_perp=base_perp,
        with_metamap=True,
        out_name=out_name,
    )
