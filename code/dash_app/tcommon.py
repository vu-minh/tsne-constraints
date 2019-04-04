"""Project-level common code or must-used global variable.
Note that, if having to use global var, wrap it in `dict`,
or better (???), write a function to return the need values.
"""


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
