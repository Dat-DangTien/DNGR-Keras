import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sklearn.cluster import KMeans
import matplotlib.colors as colors
from itertools import cycle
import time
import matplotlib.pyplot as plt
import subprocess
from utils import tsne
import pdb
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as nmi
import scipy.io as sio

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def cluster(reprsn, true_labels, methods=["kmeans"], n_clusters_range=[2, 3, 4]):
    results = []
    plot_data = defaultdict(lambda: {"n_clusters": [], "ACC": [], "NMI": []})

    def clustering_accuracy(y_true, y_pred):
        y_true = np.array(y_true).astype(np.int64)
        y_pred = np.array(y_pred).astype(np.int64)
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(len(y_pred)):
            w[y_pred[i], y_true[i]] += 1
        ind = linear_sum_assignment(w.max() - w)
        return sum([w[i, j] for i, j in zip(*ind)]) * 1.0 / len(y_pred)

    for method in methods:
        print(f"\n=== {method.upper()} ===")
        for n_clusters in n_clusters_range:
            if method == "kmeans":
                model = KMeans(n_clusters=n_clusters, random_state=42)
            elif method == "kmeans++":
                model = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            elif method == "spectral":
                model = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
            elif method == "agglomerative":
                model = AgglomerativeClustering(n_clusters=n_clusters)
            else:
                print(f"Thuật toán '{method}' chưa được hỗ trợ.")
                continue

            pred_labels = model.fit_predict(reprsn)
            acc = clustering_accuracy(true_labels, pred_labels)
            nmi = normalized_mutual_info_score(true_labels, pred_labels)

            print(f"#clusters = {n_clusters} | ACC = {acc:.4f} | NMI = {nmi:.4f}")

            results.append({
                "method": method,
                "n_clusters": n_clusters,
                "ACC": acc,
                "NMI": nmi
            })

            plot_data[method]["n_clusters"].append(n_clusters)
            plot_data[method]["ACC"].append(acc)
            plot_data[method]["NMI"].append(nmi)

    # Vẽ biểu đồ cho từng phương pháp
    for method in plot_data:
        n_clusters = plot_data[method]["n_clusters"]
        acc_values = plot_data[method]["ACC"]
        nmi_values = plot_data[method]["NMI"]

        plt.figure()
        plt.plot(n_clusters, acc_values, marker='o', label='ACC')
        plt.plot(n_clusters, nmi_values, marker='s', label='NMI')
        plt.title(f"Clustering Results - {method}")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.show()

    return results


def cluster_old(data,true_labels,n_clusters=3):

	km = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
	km.fit(data)

	km_means_labels = km.labels_
	km_means_cluster_centers = km.cluster_centers_
	km_means_labels_unique = np.unique(km_means_labels)

	colors_ = cycle(colors.cnames.keys())

	initial_dim = np.shape(data)[1]
	data_2 = tsne(data,2,initial_dim,30)

	plt.figure(figsize=(12, 6))
	plt.scatter(data_2[:,0],data_2[:,1], c=true_labels)
	plt.title('True Labels')

	return km_means_labels

data_mat = sio.loadmat('data/wine_network.mat')
labels = sio.loadmat('data/wine_label.mat')
data_mat = data_mat['adjMat']
labels = labels['wine_label']
data_edge = nx.Graph(data_mat) 

# with open('data/wine.edgelist','wb') as f:
# 	nx.write_weighted_edgelist(data_edge, f)

# subprocess.call('~/DNGR-Keras/DNGR.py --graph_type '+'undirected'+' --input '+'wine.edgelist'+' --output '+'representation',shell=True)

df = pd.read_pickle('data/representation.pkl')
reprsn = df['embedding'].values
node_idx = df['node_id'].values
reprsn = [np.asarray(row,dtype='float32') for row in reprsn]
reprsn = np.array(reprsn, dtype='float32')
true_labels = [labels[int(node)][0] for node in node_idx]
true_labels = np.asarray(true_labels, dtype='int32')
# cluster(reprsn,true_labels, n_clusters=3)
	
# plt.show()
results = cluster(
    reprsn=reprsn,
    true_labels=true_labels,
    methods=["kmeans", "kmeans++", "spectral", "agglomerative"],
    n_clusters_range=range(2, 20)
)





