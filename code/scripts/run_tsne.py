from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import joblib


embedding_dir = '../dash-app/embeddings'

X, y = load_digits(return_X_y=True)
X = StandardScaler().fit_transform(X)
X = PCA(0.95).fit_transform(X)

for perp in [5, 10, 20, 50, 100]:
    print('Run with perp=', perp)

    Z = TSNE(perplexity=perp).fit_transform(X)
    
    out_name = f"{embedding_dir}/DIGITS_perp={perp}.z"
    joblib.dump(Z, out_name)

# test read from file
in_name = f"{embedding_dir}/DIGITS_perp={5}.z"
Z2 = joblib.load(in_name)
print(Z2.shape)