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
    src='${fig_type}/${dataset_name}/${base_perp}_to_${perp}${scale}.png',
    height='30%'}
/>
"""
TD = "<td>{}</td>"
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


def html_table(tbl_id, thead, tbody):
    table = Template(TABLE)
    return table.substitute(tbl_id=tbl_id, thead=thead, tbody=tbody)


def _gen_thead():
    return "".join(
        [
            "<th>TSNE original</th>",
            "<th>TSNE original (early-stop)</th>",
            "<th>TSNE-chain original</th>",
            "<th>TSNE-chain original (early-stop)</th>",
        ]
    )


def _gen_row(perp):
    imgs = [
        FIG_IMG.format(
            idx=f"img_{perp}_normal",
            fig_type="normal",
            dataset_name=dataset_name,
            perp=perp,
        )
    ]


def gen_embedding_figures_table():
    return html_table(tbl_id="embedding_figures", thead=_gen_thead(), tbody="")


def gen_page(template_name, out_name, base_perp):
    with app.app_context():
        rendered = render_template(
            template_name,
            title="tSNE chain",
            base_perp=base_perp,
            dataset_name=dataset_name,
            embedding_figures=gen_embedding_figures_table(),
        )

    with open(out_name, "w") as out_file:
        out_file.write(rendered)
    print(f"Write to {out_name}")


hyper_params = {"FASHION200": {"base_perps": [10, 20, 30, 40]}}


if __name__ == "__main__":
    dataset_name = "FASHION200"
    _, X, _ = dataset.load_dataset(dataset_name)
    max_perp = X.shape[0] // 3
    template_name = "view_chain2.template"
    is_earlystop = "_earlystop"  # or ""

    base_perp = 40
    out_name = f"./html/{dataset_name}_base{base_perp}{is_earlystop}.html"
    gen_page(template_name, out_name, base_perp=base_perp)

    # for base_perp in hyper_params[dataset_name]["base_perp"]:
    #     for fig_scale in ["", "_autoscale"]:
    #         out_name = f"html/{dataset_name}_base{base_perp}{is_earlystop}.html"
    #         # get_kl_data()
    #         gen_page(template_name, out_name)
