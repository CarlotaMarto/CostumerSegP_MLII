import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.base import clone
from utils import (
    plot_dendrogram,
    plot_2d_with_classes,
    visualize_dimensionality_reduction
)

def plot_elbow(X_scaled, k_range=15):
    """
    Calculates and plots the Elbow Method (Inertia)
    
    Parameters:
    X_scaled: The scaled feature matrix.
    k_range: The maximum number of clusters to test (default is 15).
    """
    inertias = []

    for k in range(2, k_range + 1):
        experimental_kmeans = KMeans(n_clusters=k, random_state=42).fit(X_scaled)
        inertias.append(experimental_kmeans.inertia_)

    plt.plot(range(2, k_range + 1), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Dispersion (inertia)')
    plt.title('Elbow Method for Optimal K')
    plt.show()


def plot_silhouette(X_scaled, K_range=15):
    scores = []

    for k in range(2,K_range+1):
        experimental_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = experimental_kmeans.fit_predict(X_scaled)
        scores.append(silhouette_score(X_scaled, labels))
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(2,K_range+1), scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Scores')
    plt.show()

def print_silhouette_scores(X_scaled, k_values=[6, 7, 8, 9, 10]):
    """
    Iterates through a list of K values, trains a K-Means model for each, 
    and prints the corresponding Silhouette Score.
    
    Parameters:
    X_scaled: The scaled feature matrix.
    k_values: A list of integers representing the number of clusters to test.
    """
    print("Evaluating Silhouette Scores:")
    
    for k in k_values:
        experimental_kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = experimental_kmeans.fit_predict(X_scaled)

        # Using sample_size to ensure fast execution without crashing
        score = silhouette_score(X_scaled, labels, sample_size=5000, random_state=42)

        print(f"k={k} | silhouette={score:.4f}")



def apply_kmeans_perc(X_scaled, dataset, X, k=8):
    """
    Fits a K-Means model to the scaled data, appends the cluster labels 
    to both the original dataset and the unscaled X matrix, and returns the model.

    Parameters:
    X_scaled: The scaled feature matrix used for training.
    dataset: The original complete dataset.
    X: The unscaled feature matrix.
    k: The number of clusters to use (default is 8).
    
    Returns:
    kmeans_model: The trained KMeans object.
    """

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
 
    dataset['cluster_kmeans_perc'] = labels
    X['cluster_kmeans_perc'] = labels
    X_scaled['cluster_kmeans_perc'] = labels
 
    print(f"Customer Distribution for K={k}:")
    print(dataset["cluster_kmeans_perc"].value_counts().sort_index())
    
    return kmeans

def apply_kmeans_lifetime(X_scaled, dataset, X, k=8):
    """
    Fits a K-Means model to the scaled data, appends the cluster labels 
    to both the original dataset and the unscaled X matrix, and returns the model.

    Parameters:
    X_scaled: The scaled feature matrix used for training.
    dataset: The original complete dataset.
    X: The unscaled feature matrix.
    k: The number of clusters to use (default is 8).
    
    Returns:
    kmeans_model: The trained KMeans object.
    """

    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
 
    dataset['cluster_kmeans_lifetime'] = labels
    X['cluster_kmeans_lifetime'] = labels
    X_scaled['cluster_kmeans_lifetime'] = labels
 
    print(f"Customer Distribution for K={k}:")
    print(dataset["cluster_kmeans_lifetime"].value_counts().sort_index())
    
    return kmeans

def cluster_groupby(X, cluster_kmeans, method):

    if method == 'mean':
        grouped = X.groupby(cluster_kmeans).mean().T
    elif method == 'median':
        grouped = X.groupby(cluster_kmeans).median().T
    
    print(f"Grouped by {method}")
    
    return grouped