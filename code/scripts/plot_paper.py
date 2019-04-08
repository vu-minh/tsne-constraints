import os
import joblib
import numpy as np
import argparse

from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from common.dataset import dataset
from common.plot.scatter import scatter_with_box, annotate_text, config_font_size
from icommon import hyper_params
from matplotlib.ticker import FuncFormatter

# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler, StandardScaler


def plot_metamap(
    ax, all_perps=[], base_perp=None, highlight_selected_perps=[], title=""
):
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
    all_losses = []
    highlight_idx = {}

    for idx, perp in enumerate(all_perps):
        in_name = f"{embedding_dir}/{in_name_prefix}{perp}{earlystop}.z"
        data = joblib.load(in_name)
        all_Z.append(data["embedding"].ravel())
        all_losses.append(data["kl_divergence"])
        if perp in highlight_selected_perps:
            highlight_idx[idx] = perp

    if len(all_Z) < 1:
        raise ValueError("Not enough embeddings to make metaplot")

    # all_Z = StandardScaler().fit_transform(all_Z)
    meta_Z = TSNE(perplexity=10, random_state=2019).fit_transform(all_Z)

    ax.grid(False)
    sct = ax.scatter(meta_Z[:, 0], meta_Z[:, 1], c=all_perps, cmap="inferno")
    ax.set_title(f"{dataset_name}{out_name_sufix}")

    # show colorbar beside the metaplot
    cbar = plt.colorbar(sct, aspect=40, pad=0.04)
    cbar.ax.set_title("Perplexity")

    # highlight selected perplexities
    if len(highlight_idx) > 0:
        highlight_pos = meta_Z[list(highlight_idx.keys())]
        scatter_with_box(ax, all_pos=highlight_pos)
        # add annotation text above the box
        for idx, selected_perp in enumerate(highlight_idx.values()):
            annotate_text(ax, text=selected_perp, pos=highlight_pos[idx])

    # highlight base perplexity
    base_perp_idx = all_perps.index(base_perp)
    scatter_with_box(
        ax, all_pos=np.array([meta_Z[base_perp_idx, :]]), marker="h", color="red"
    )
    annotate_text(
        ax, base_perp, pos=meta_Z[base_perp_idx], offset=(-18, -24), text_color="red"
    )

    plt.tight_layout()


def plot_an_embedding(ax, perp, base_perp=None, labels=None, disable_ticks=False):
    if base_perp is None:
        embedding_dir = f"{dir_path}/normal/{dataset_name}"
        in_name_prefix = ""
    else:
        embedding_dir = f"{dir_path}/chain/{dataset_name}"
        in_name_prefix = f"{base_perp}_to_"

    in_name = f"{embedding_dir}/{in_name_prefix}{perp}{earlystop}.z"
    data = joblib.load(in_name)
    Z = data["embedding"]
    ax.scatter(Z[:, 0], Z[:, 1], c=labels, alpha=0.5, cmap="jet")
    ax.grid(False)
    ax.set_title(f"perplexity = {perp}")

    if disable_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


def plot_metamap_with_some_perps(
    selected_perps=[], base_perp=30, with_metamap=True, out_name="metaplot", labels=None
):
    """Plot examples of chain-tsne with perp in `perps`.
	If `with_metamap` is set, the metamap of all chain-tsne will be shown.
	"""
    (n_rows, n_cols) = (2, 5 if with_metamap else 3)
    plt.figure(figsize=(4 * n_cols, 5 * n_rows))

    _, X, labels = dataset.load_dataset(dataset_name)
    all_perps = range(3, X.shape[0] // 3)

    # plot metaplot
    if with_metamap:
        ax0 = plt.subplot2grid((n_rows, n_cols), (0, 3), rowspan=2, colspan=2)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        plot_metamap(
            ax=ax0,
            all_perps=all_perps,
            base_perp=base_perp,
            highlight_selected_perps=selected_perps,
        )

    # plot embeddings corresponding to the selected perplexities
    for i, perp in enumerate(selected_perps):
        row_i, col_i = i // 3, i % 3
        ax_i = plt.subplot2grid((n_rows, n_cols), (row_i, col_i))
        plot_an_embedding(
            ax=ax_i, perp=perp, base_perp=base_perp, labels=labels, disable_ticks=True
        )

    plt.tight_layout()
    plt.savefig(f"{out_name}.{OUTPUT_EXT}")


def plot_running_time(
    ax, dataset_name, base_perp, show_legend=False, perp_in_log_scale=False
):
    _, X, _y = dataset.load_dataset(dataset_name)
    all_perps = range(1, X.shape[0] // 3)
    running_time_normals = []
    running_time_chains = []
    n_iter_normals = []
    n_iter_chains = []

    embedding_dir_normal = f"{dir_path}/normal/{dataset_name}"
    embedding_dir_chain = f"{dir_path}/chain/{dataset_name}"

    # prepare data: read csv for all perplexities of one dataset
    for perp in all_perps:
        in_name_normal = f"{embedding_dir_normal}/{perp}{earlystop}.z"
        data_normal = joblib.load(in_name_normal)
        running_time_normals.append(data_normal["running_time"])
        n_iter_normals.append(data_normal["n_iter"])

        in_name_chain = f"{embedding_dir_chain}/{base_perp}_to_{perp}{earlystop}.z"
        data_chain = joblib.load(in_name_chain)
        running_time_chain = data_chain["running_time"] if perp != base_perp else np.nan
        running_time_chains.append(running_time_chain)
        n_iter_chains.append(data_chain["n_iter"])

    # plot running time for one dataset
    if perp_in_log_scale:
        ax.semilogx(
            all_perps, running_time_normals, lw=2.0, label="tSNE normal", basex=np.e
        )
        ax.semilogx(
            all_perps, running_time_chains, lw=2.0, label="chain-tSNE", basex=np.e
        )
        ax.set_xscale("log", basex=np.e)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, pos: int(value)))
        ax.set_xtitle("perplexity in log-scale")
    else:
        ax.plot(all_perps, running_time_normals, lw=2.0, label="tSNE normal")
        ax.plot(all_perps, running_time_chains, lw=2.0, label="chain-tSNE")
        ax.set_xlabel("perplexity")

    ax.set_title(dataset_name)
    if show_legend:
        ax.set_ylabel("Running time (second)")
        plt.legend()


def plot_running_time_all_datasets(dataset_names, base_perp, out_name):
    (n_rows, n_cols) = (1, len(dataset_names))
    plt.figure(figsize=(8 * n_cols, 6 * n_rows))

    for i, dataset_name in enumerate(dataset_names):
        ax_i = plt.subplot2grid((n_rows, n_cols), (0, i))
        plot_running_time(ax_i, dataset_name, base_perp, show_legend=(i == 0))

    plt.tight_layout()
    plt.savefig(f"{out_name}.{OUTPUT_EXT}")


if __name__ == "__main__":
    # plt.rcParams.update({"font.size": 22})
    config_font_size(min_size=16)

    OUTPUT_EXT = "png"  # "pdf" for paper
    FIG_DIR = "../../figures"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    data_dir = f"{dir_path}/data"
    dataset.set_data_home(data_dir)

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

    def _plot_metamap():
        out_name = f"{FIG_DIR}/metaplot_{dataset_name}_base{base_perp}"
        plot_metamap_with_some_perps(
            selected_perps=hyper_params[dataset_name]
            .get("selected_perps", {})
            .get(base_perp, []),
            base_perp=base_perp,
            with_metamap=True,
            out_name=out_name,
        )

    def _plot_running_time():
        out_name = f"{FIG_DIR}/runningtime_base{base_perp}"
        dataset_names = ["BREAST_CANCER", "COIL20", "FASHION200", "COUNTRY2014"]
        plot_running_time_all_datasets(dataset_names, base_perp, out_name=out_name)

    def _plot_examples():
        from itertools import product

        _, _, labels = dataset.load_dataset(dataset_name)
        base_perps = [None, 30]
        selected_perps = [5, 40]

        for base_perp, selected_perp in product(base_perps, selected_perps):
            name_prefix = "" if base_perp is None else f"base{base_perp}"
            out_name = f"{FIG_DIR}/eg_{dataset_name}{name_prefix}_perp{selected_perp}"

            _, ax = plt.subplots(1, 1, figsize=(6, 6))
            plot_an_embedding(ax, selected_perp, base_perp, labels=labels)
            plt.tight_layout()
            plt.savefig(f"{out_name}.{OUTPUT_EXT}")

    _plot_metamap()
    _plot_running_time()
    _plot_examples()
