"""Project-level common code or must-used global variable.
Note that, if having to use global var, wrap it in `dict`,
or better (???), write a function to return the need values.
"""

import joblib
import dash_html_components as html

STATIC_IMAGE_DIR = "./static/svg"
LINK_DIR = "./links"


def colorize_link(link_type):
    """Return css color for a link,
    e.g. "green" for sim-link, "red" for dissim-link, "orange" for other type

    :link_type: type of the link, can be "sim-", "dissim-", or other type
    :returns: css color
    """
    if link_type.lower().startswith("sim"):
        return "green"
    elif link_type.lower().startswith("dis"):
        return "red"
    else:
        return "orange"


def get_image_url(dataset_name, img_id, cmap_type="gray"):
    """Return URL for image of a datapoint `img_id` from a stacked svg"""
    return (f"{STATIC_IMAGE_DIR}/{dataset_name}_{cmap_type}.svg#{img_id}",)


def get_image_elem(dataset_name, img_id, cmap_type="gray", width="100px"):
    """Generate dash html image element"""

    return html.Img(
        src=get_image_url(dataset_name, img_id, cmap_type), width=f"{width}"
    )


def get_auto_generated_constraints(dataset_name, n_sim=50, n_dis=50):
    in_name = f"{LINK_DIR}/auto_{dataset_name}_{n_sim}sim_{n_dis}dis.pkl"
    links = joblib.load(in_name)
    assert len(links) == n_sim + n_dis
    return links
