"""Run metrics on the pre-calculated embeddings
"""
import os
import joblib
import numpy as np
import pandas as pd
from common.dataset import dataset
from common.metric.dr_metrics import DRMetric


dir_path = os.path.dirname(os.path.realpath(__file__))
data_dir = f"{dir_path}/data"
embedding_dir = f"{dir_path}/embeddings"
metric_dir = f"{dir_path}/metrics"

dataset.set_data_home(data_dir)
if not os.path.exists(metric_dir):
    os.mkdir(metric_dir)


def get_list_embeddings(dataset_name):
    target_dir = os.path.join(embedding_dir, dataset_name)
    return [os.path.join(embedding_dir, dataset_name, f)
            for f in os.listdir(target_dir) if f.endswith('.z')]


def calculate_metrics(dataset_name, metric_names):
    _, X, _ = dataset.load_dataset(dataset_name)
    N = X.shape[0]

    results = []
    for in_file_name in get_list_embeddings(dataset_name):
        perp = int(in_file_name.split('/')[-1][:-2])
        data = joblib.load(in_file_name)

        # Recall: data = dict(
        #     running_time=running_time,
        #     embedding=tsne.embedding_,
        #     kl_divergence=tsne.kl_divergence_,
        #     n_jobs=tsne.n_jobs if USE_MULTICORE else 1,
        #     n_iter=tsne.n_iter_,
        #     learning_rate=tsne.learning_rate,
        #     random_state=tsne.random_state,
        # )

        data['perplexity'] = perp
        data['bic'] = 2 * data['kl_divergence'] + np.log(N) * perp / N

        drMetric = DRMetric(X, data['embedding'])
        for metric_name in metric_names:
            metric_method = getattr(drMetric, metric_name)
            data[metric_name] = metric_method()
        # do not need the embedding field
        del data['embedding']
        results.append(data)

    write_metric_results(dataset_name, results)


def write_metric_results(dataset_name, results):
    header = ','.join(results[0].keys())
    body = [','.join(map(str, row.values())) for row in results]
    with open(f"{metric_dir}/{dataset_name}.csv", 'w') as csv_file:
        csv_file.write(header)
        csv_file.write('\n')
        csv_file.write('\n'.join(body))


def test_metric_results(dataset_name):
    df = pd.read_csv(f"{metric_dir}/{dataset_name}.csv",
                     index_col='perplexity')
    print(df)


if __name__ == '__main__':
    dataset_name = 'FASHION500'
    metric_names = [
        'auc_rnx',
        'pearsonr',
        'mds_isotonic',
        'cca_stress',
        'sammon_nlm'
    ]
    calculate_metrics(dataset_name, metric_names)
    test_metric_results(dataset_name)
