hyper_params = {
    "COIL20": {
        "early_stop_conditions": {
            "n_iter_without_progress": 200,
            "min_grad_norm": 1e-10,  # perp=200: 5e-5, perp=300: 1e-10
        },
        "base_perps": [20, 30, 50, 75],
    },
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
    "BREAST_CANCER": {
        "early_stop_conditions": {
            "n_iter_without_progress": 120,
            "min_grad_norm": 2e-04,
        },
        "base_perps": [10, 20, 30, 50],
    },
    "MPI": {
        "early_stop_conditions": {
            "n_iter_without_progress": 50,
            "min_grad_norm": 5e-04,
        },
        "base_perps": [5, 10, 20, 30],
    },
    "DIABETES": {
        "early_stop_conditions": {
            "n_iter_without_progress": 100,
            "min_grad_norm": 2e-04,
        },
        "base_perps": [10, 20, 30, 40],
    },
    "COUNTRY2014": {
        "early_stop_conditions": {
            "n_iter_without_progress": 50,
            "min_grad_norm": 2e-04,
        },
        "base_perps": [5, 10, 20, 30],
    },
}
