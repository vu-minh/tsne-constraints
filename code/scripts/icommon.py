hyper_params = {
    "DIGITS": {
        "early_stop_conditions": {
            "n_iter_without_progress": 150,
            "min_grad_norm": 1e-04,
        },
        "base_perps": [20, 30, 50, 75],
    },
    "FASHION200": {
        "early_stop_conditions": {
            "n_iter_without_progress": 120,
            "min_grad_norm": 5e-04,
        },
        "base_perps": [10, 20, 30, 40],
    },
}
