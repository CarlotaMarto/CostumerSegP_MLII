import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram

def load_local_data(filename="info_clustering_ready.csv"):
    """Carrega o ficheiro diretamente a partir da pasta de caminhos relativos."""
    parent_dir = os.path.dirname(os.getcwd())
    path_datasets = os.path.join(parent_dir, "datasets", filename)
    return pd.read_csv(path_datasets)

def run_kmeans_pipeline(df, exclude_cols, k):
    """Executa o KMeans removendo as colunas indicadas."""
    features = df.drop(columns=exclude_cols, errors='ignore')
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_
    
    sample_size = min(5000, len(features))
    score = silhouette_score(features, labels, sample_size=sample_size, random_state=42)
    
    return labels, centroids, score

def plot_elbow_and_silhouette(df, exclude_cols, max_k=10):
    """Gera os graficos de avaliacao de K."""
    features = df.drop(columns=exclude_cols, errors='ignore')
    inertias = []
    silhouettes = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(features)
        inertias.append(km.inertia_)
        
        sample_size = min(3000, len(features))
        silhouettes.append(silhouette_score(features, labels, sample_size=sample_size, random_state=42))
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    ax1.plot(k_range, inertias, 'bo-', color='#0047AB')
    ax1.set_title('Elbow Method (Inertia)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia')
    ax1.grid(True)
    
    ax2.plot(k_range, silhouettes, 'ro-', color='#D9534F')
    ax2.set_title('Silhouette Analysis', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Average Silhouette Score')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_pca_clusters(df, exclude_cols, labels, title="PCA Cluster Visualization"):
    """Visualizacao bidimensional via PCA."""
    features = df.drop(columns=exclude_cols, errors='ignore')
    
    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=labels, cmap='tab10', alpha=0.6, s=15)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
    plt.colorbar(scatter, label='Cluster ID')
    plt.grid(True)
    plt.show()

def plot_linkage_dendrogram(centroids, title="Hierarchical Clustering on KMeans Centroids"):
    """Gera o dendrograma dos centroides."""
    linked = linkage(centroids, method='ward')
    
    plt.figure(figsize=(10, 6))
    dendrogram(linked, labels=range(len(centroids)), distance_sort='descending', show_leaf_counts=True)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('KMeans Cluster ID')
    plt.ylabel('Distance (Ward Linkage)')
    plt.grid(True)
    plt.show()
    return linked


def get_color_map(labels, cmap_name='tab10'):
    unique_labels = sorted(np.unique(labels))
    color_map = {label: cm.get_cmap(cmap_name)(i % 10) for i, label in enumerate(unique_labels)}
    return [color_map[label] for label in labels]


def run_kmeans(df, k, exclude_cols=None, random_state=42):
    """
    Runs KMeans clustering on the input DataFrame, excluding specified columns.

    Returns:
        labels: cluster assignments
        centroids: cluster centers
        score: silhouette score
    """
    if exclude_cols is None:
        exclude_cols = []

    features = df.drop(columns=exclude_cols).values
    model = KMeans(n_clusters=k, random_state=random_state)
    labels = model.fit_predict(features)
    centroids = model.cluster_centers_
    score = silhouette_score(features, labels)
    return labels, centroids, score


def elbow_and_silhouette(df, max_k=10, exclude_cols=None, random_state=42):
    """
    Plots Elbow and Silhouette score curves for a range of k values.
    
    Returns:
        inertias: list of inertia values
        silhouettes: list of silhouette scores
    """
    if exclude_cols is None:
        exclude_cols = []

    data = df.drop(columns=exclude_cols).values
    inertias = []
    silhouettes = []

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        labels = kmeans.fit_predict(data)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(data, labels)
        silhouettes.append(score)
        print(f"K={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={score:.4f}")

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel("Number of clusters (k)")
    ax1.set_ylabel("Inertia", color="tab:blue")
    ax1.plot(range(2, max_k + 1), inertias, marker='o', color="tab:blue", label="Inertia")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:green")
    ax2.plot(range(2, max_k + 1), silhouettes, marker='x', linestyle='--', color="tab:green", label="Silhouette")
    ax2.tick_params(axis='y', labelcolor="tab:green")

    plt.title("Elbow and Silhouette Scores")
    fig.tight_layout()
    plt.show()

    return inertias, silhouettes




def visualize_umap(embedding, labels=None, title='UMAP Projection', xlabel='Component 1', ylabel='Component 2', cmap_name='tab10'):
    plt.figure(figsize=(10, 8))

    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=10)
    else:
        unique_labels = sorted(np.unique(labels))
        colors = get_color_map(labels, cmap_name)

        for label in unique_labels:
            idx = labels == label
            color = colors[np.where(labels == label)[0][0]]
            plt.scatter(embedding[idx, 0], embedding[idx, 1], c=[color], s=10, label=f"Cluster {label}")

        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
