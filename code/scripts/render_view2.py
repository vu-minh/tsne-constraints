# create new table view for tSNE embeddings
# 4 columns:
# TSNE original | TSNE original early-stop | TSNE chain early-stop | TSNE chain
# Each row is the embeddings for each perplexity


import os
import flask
import pandas as pd
from flask import render_template
from string import Template
from common.dataset import dataset
from icommon import hyper_params


dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{dir_path}/data"
dataset.set_data_home(data_dir)

app = flask.Flask("render_view_app")


CSV_PATH = "./plot_chain"

TABLE = """
<table id="${tbl_id}">
    <thead> ${thead} </thead>
    <tbody> ${tbody} </tbody>
</table>
"""

FIG_IMG = """
<img id='img_${idx}' class='img-thumbnail' alt='xxx'
    src='../${fig_type}/${dataset_name}/${img_name}'
/>
"""

TH = "<th align='center' width='22%'>{}</th>"
TD = "<td align='center' width='22%'>{}</td>"
TDP = "<td><p>{}</p></td>"
TR = "<tr>{}</tr>"
TRC = '<tr style="color:{color}">{data}</tr>'

kl_data = {"kl_chain_normal": {}, "kl_chain": {}, "kl_normal": {}}


def get_kl_data():
    data_path = f"{CSV_PATH}/{dataset_name}"
    df_chain = pd.read_csv(f"{data_path}_chain_{base_perp}.csv", index_col="perplexity")
    df_normal = pd.read_csv(
        f"{data_path}_normal_{base_perp}.csv", index_col="perplexity"
    )
    df_compare = pd.read_csv(f"{data_path}_{base_perp}.csv", index_col="perplexity")
    kl_data["kl_chain_normal"] = df_compare["kl_chain_normal"].to_dict()
    kl_data["kl_chain"] = df_chain["kl_Qbase_Q"].to_dict()
    kl_data["kl_normal"] = df_normal["kl_Qbase_Q"].to_dict()


def html_table(tbl_id="", thead="", tbody=""):
    table = Template(TABLE)
    return table.substitute(tbl_id=tbl_id, thead=thead, tbody=tbody)


def _gen_thead():
    return "".join(
        [
            "<th width='10%'>perp</th>",
            TH.format("TSNE-original"),
            TH.format("TSNE-original (early-stop)"),
            TH.format("TSNE-chain (early-stop)"),
            TH.format("TSNE-chain"),
        ]
    )


def _gen_row(perp, base_perp):
    imgs = [
        Template(FIG_IMG).substitute(
            idx=f"img_{perp}",
            fig_type="normal",
            dataset_name=dataset_name,
            img_name=f"{perp}_all.png",
        ),
        Template(FIG_IMG).substitute(
            idx=f"img_{perp}",
            fig_type="normal",
            dataset_name=dataset_name,
            img_name=f"{perp}_earlystop_all.png",
        ),
        Template(FIG_IMG).substitute(
            idx=f"img_{base_perp}-{perp}",
            fig_type="chain",
            dataset_name=dataset_name,
            img_name=f"{base_perp}_to_{perp}_earlystop_all.png",
        ),
        Template(FIG_IMG).substitute(
            idx=f"img_{base_perp}-{perp}",
            fig_type="chain",
            dataset_name=dataset_name,
            img_name=f"{base_perp}_to_{perp}_all.png",
        ),
    ]
    return TR.format(
        "".join([f"<td width='10%'>{perp}</td>"] + [TD.format(img) for img in imgs])
    )


def _gen_tbody(run_range, base_perp):
    return "\n".join([_gen_row(perp, base_perp) for perp in run_range])


def gen_page(template_name, out_name, run_range, base_perp):
    with app.app_context():
        rendered = render_template(
            template_name,
            title="tSNE chain",
            base_perp=base_perp,
            dataset_name=dataset_name,
            table_header=html_table(thead=_gen_thead()),
            embedding_figures=html_table(tbody=_gen_tbody(run_range, base_perp)),
        )

    with open(out_name, "w") as out_file:
        out_file.write(rendered)
    print(f"Write to {out_name}")


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

    _, X, _ = dataset.load_dataset(dataset_name)
    run_range = (
        [test_perp - 1, test_perp, test_perp + 1] if DEV else range(2, X.shape[0] // 3)
    )
    template_name = "view_chain2.template"

    for base_perp in hyper_params[dataset_name]["base_perps"]:
        out_name = f"html/{dataset_name}_base{base_perp}_v2.html"
        gen_page(template_name, out_name, run_range=run_range, base_perp=base_perp)
