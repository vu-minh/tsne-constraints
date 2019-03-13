import flask
import pandas as pd
from flask import render_template
from string import Template

app = flask.Flask("render_view_app")


DATA_PATH = "../"
CSV_PATH = "./plot_chain"

TABLE = """
<table id=${tbl_id}>
    <thead> ${thead} </thead>
    <tbody> ${tbody} </tbody>
</table>
"""

FIG_IMG = """
<img id='img_${idx}' class='img-thumbnail' alt='xxx'
    src='${data_path}/${fig_type}/${dataset_name}/${base_perp}_to_${perp}${scale}.png',
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


def gen_fig(fig_type, perp):
    return Template(FIG_IMG).substitute(
        idx=f"base_{perp}",
        data_path=DATA_PATH,
        fig_type=fig_type,
        dataset_name=dataset_name,
        base_perp=base_perp,
        perp=perp,
        scale=fig_scale,
    )


def gen_normal_figures():
    tbody = "\n".join(
        [
            TR.format(TDP.format("(Q_base)")),
            TR.format(
                "\n".join(
                    [TD.format(gen_fig("normal", base_perp)), TD.format("tSNE chain")]
                )
            ),
            TR.format(TDP.format("Compare tSNE chain & normal")),
            TR.format(
                "\n".join(
                    [TD.format(gen_fig("normal", base_perp)), TD.format("tSNE normal")]
                )
            ),
            TR.format(TDP.format("(Q_base)")),
        ]
    )
    thead = f"<th>Base perplexity = {base_perp}</th>"
    return html_table(tbl_id="tbl_base", thead=thead, tbody=tbody)


def gen_chain_figures():
    perp_range = range(2, max_perp + 1)
    row_fig_chains = [TD.format(gen_fig("chain", p)) for p in perp_range]
    row_fig_normals = [TD.format(gen_fig("normal", p)) for p in perp_range]

    row_kl_chains = [
        TDP.format(
            "KL[Q_base||Q_chain] = {:.3f}".format(kl_data["kl_chain"].get(p, -1))
        )
        for p in perp_range
    ]

    row_kl_normals = [
        TDP.format(
            "KL[Q_base||Q_normal] = {:.3f}".format(kl_data["kl_normal"].get(p, -1))
        )
        for p in perp_range
    ]

    row_kl_chain_normal = [
        TDP.format(
            "KL[Q_chain||Q_normal] = {:.3f}".format(
                kl_data["kl_chain_normal"].get(p, -1)
            )
        )
        for p in perp_range
    ]

    tbody = "\n".join(
        [
            TRC.format(color="#ff4b0d", data="\n".join(row_kl_chains)),
            TR.format("\n".join(row_fig_chains)),
            TRC.format(color="black", data="\n".join(row_kl_chain_normal)),
            TR.format("\n".join(row_fig_normals)),
            TRC.format(color="blue", data="\n".join(row_kl_normals)),
        ]
    )
    thead = "".join(map(lambda p: f"<th>Perplexity = {p}</th>", perp_range))
    return html_table(tbl_id="tbl_chains", thead=thead, tbody=tbody)


def gen_page(template_name, out_name):
    with app.app_context():
        rendered = render_template(
            template_name,
            title="tSNE chain",
            base_perp=base_perp,
            normal_figures=gen_normal_figures(),
            chain_figures=gen_chain_figures(),
            dataset_name=dataset_name,
        )

    with open(out_name, "w") as out_file:
        out_file.write(rendered)


if __name__ == "__main__":
    dataset_name = "DIGITS"
    max_perp = 600
    template_name = "view_chain.template"

    for base_perp in [10, 20, 25, 30, 40, 50, 75, 100, 200]:
        for fig_scale in ["", "_autoscale"]:
            out_name = f"html/{dataset_name}_base{base_perp}{fig_scale}.html"
            get_kl_data()
            gen_page(template_name, out_name)
