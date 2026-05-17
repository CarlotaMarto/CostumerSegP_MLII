"""
utils_clustering.py
Clustering utilities for customer segmentation

Author: Data Science Team
Version: 2.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from sklearn.metrics import pairwise_distances_argmin_min
import warnings
warnings.filterwarnings('ignore')

# Optional import for UMAP
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


# ============================================================================
# DATA LOADING
# ============================================================================

def load_local_data(filepath, index_col=None):
    """
    Load customer data from CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        if index_col and index_col in df.columns:
            df.set_index(index_col, inplace=True)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return pd.DataFrame()


def load_outliers(filepath="outlier_dataset.csv"):
    """
    Load outlier dataset.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Outliers loaded: {df.shape[0]} rows")
        return df
    except FileNotFoundError:
        print(f"No outlier file found at: {filepath}")
        return pd.DataFrame()


# ============================================================================
# K-MEANS CLUSTERING
# ============================================================================

def run_kmeans(df, k, exclude_cols=None, random_state=42):
    """
    Run KMeans clustering on pre-scaled data.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    score = silhouette_score(X, labels)
    
    print(f"KMeans completed: k={k}, silhouette={score:.4f}")
    return labels, centroids, score


# ============================================================================
# ELBOW AND SILHOUETTE ANALYSIS
# ============================================================================

def elbow_and_silhouette(df, max_k=10, exclude_cols=None, random_state=42):
    """
    Plots Elbow and Silhouette score curves for a range of k values.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    inertias = []
    silhouettes = []
    k_values = range(2, max_k + 1)
    
    print("Computing metrics for k from 2 to", max_k)
    print("-" * 50)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        score = silhouette_score(X, labels)
        silhouettes.append(score)
        print(f"k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={score:.4f}")
    
    results_df = pd.DataFrame({
        'k': list(k_values),
        'inertia': inertias,
        'silhouette': silhouettes
    })
    
    # Plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    ax1.set_xlabel("Number of clusters (k)", fontsize=12)
    ax1.set_ylabel("Inertia", color="tab:blue", fontsize=12)
    ax1.plot(k_values, inertias, marker='o', color="tab:blue", linewidth=2, markersize=8)
    ax1.tick_params(axis='y', labelcolor="tab:blue")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel("Silhouette Score", color="tab:green", fontsize=12)
    ax2.plot(k_values, silhouettes, marker='x', linestyle='--', color="tab:green", linewidth=2, markersize=8)
    ax2.tick_params(axis='y', labelcolor="tab:green")
    
    plt.title("Elbow Method and Silhouette Score", fontsize=14, fontweight='bold')
    
    # Highlight best silhouette (excluding k=2 if needed)
    best_idx = np.argmax(silhouettes[1:]) + 1  # Skip k=2 if desired
    best_k = k_values[best_idx]
    ax2.axvline(x=best_k, color='red', linestyle='--', alpha=0.5, label=f'Best k={best_k}')
    
    fig.tight_layout()
    plt.show()
    
    return inertias, silhouettes, results_df


# ============================================================================
# PCA VISUALIZATION (IMPROVED)
# ============================================================================

def visualize_pca_with_variance(df, exclude_cols=None, labels=None, title='PCA - Cluster Visualization'):
    """
    Visualize clusters using PCA with explained variance display.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    pca = PCA(n_components=2)
    pca_embedding = pca.fit_transform(X)
    
    # Calculate explained variance
    var_ratio = pca.explained_variance_ratio_
    total_var = var_ratio.sum() * 100
    
    plt.figure(figsize=(14, 10))
    
    if labels is None:
        plt.scatter(pca_embedding[:, 0], pca_embedding[:, 1], s=30, alpha=0.6, c='steelblue')
    else:
        unique_labels = sorted(np.unique(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(pca_embedding[mask, 0], pca_embedding[mask, 1], 
                       c=[color], s=30, alpha=0.6, label=f'Cluster {label}')
        
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.xlabel(f'PC1 ({var_ratio[0]:.1%} variance)', fontsize=12)
    plt.ylabel(f'PC2 ({var_ratio[1]:.1%} variance)', fontsize=12)
    
    # Add note about total variance
    note = f'Total variance explained by PC1 + PC2: {total_var:.1f}%'
    if total_var < 50:
        note += '\nNote: Low explained variance suggests clusters exist in higher dimensions'
    
    plt.title(f'{title}\n{note}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return pca_embedding, var_ratio, total_var


# ============================================================================
# UMAP VISUALIZATION (IMPROVED)
# ============================================================================

def visualize_umap_clusters(df, exclude_cols=None, labels=None, 
                            n_neighbors=15, min_dist=0.1, random_state=42,
                            title='UMAP - Cluster Visualization'):
    """
    Visualize clusters using UMAP dimensionality reduction.
    """
    if not UMAP_AVAILABLE:
        print("UMAP not available. Install with: pip install umap-learn")
        return None
    
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    X = df[feature_cols].values
    
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, 
                        random_state=random_state, n_components=2)
    embedding = reducer.fit_transform(X)
    
    plt.figure(figsize=(14, 10))
    
    if labels is None:
        plt.scatter(embedding[:, 0], embedding[:, 1], s=30, alpha=0.6, c='steelblue')
    else:
        unique_labels = sorted(np.unique(labels))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            mask = labels == label
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=[color], s=30, alpha=0.6, label=f'Cluster {label}')
        
        plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return embedding


# ============================================================================
# SILHOUETTE PLOT
# ============================================================================

def silhouette_plot(X, cluster_labels, title='Silhouette Plot'):
    """
    Create detailed silhouette plot for cluster validation.
    """
    silhouette_avg = silhouette_score(X, cluster_labels)
    sample_silhouette_values = silhouette_samples(X, cluster_labels)
    
    unique_labels = sorted(np.unique(cluster_labels))
    n_clusters = len(unique_labels)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    y_lower = 10
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    for i, label in enumerate(unique_labels):
        ith_values = sample_silhouette_values[cluster_labels == label]
        ith_values.sort()
        size_i = ith_values.shape[0]
        y_upper = y_lower + size_i
        
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_values,
                         facecolor=colors[i], edgecolor=colors[i], alpha=0.7)
        
        ax.text(-0.05, y_lower + 0.5 * size_i, str(label))
        y_lower = y_upper + 10
    
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
               label=f"Average silhouette = {silhouette_avg:.3f}")
    ax.set_title(f'{title} - Average Score: {silhouette_avg:.3f}', fontsize=14, fontweight='bold')
    ax.set_xlabel("Silhouette Coefficient Values", fontsize=12)
    ax.set_ylabel("Cluster Label", fontsize=12)
    ax.legend(loc='upper right')
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    
    return silhouette_avg


# ============================================================================
# HIERARCHICAL CLUSTERING ON CENTROIDS
# ============================================================================

def plot_linkage_dendrogram(centroids, title="Hierarchical Clustering on K-Means Centroids", 
                            method='ward', figsize=(10, 6)):
    """
    Plot dendrogram from cluster centroids to show cluster relationships.
    """
    Z = linkage(centroids, method=method)
    
    plt.figure(figsize=figsize)
    dendrogram(Z, labels=[f'Cluster {i}' for i in range(len(centroids))], 
               leaf_rotation=45, leaf_font_size=10)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Cluster", fontsize=12)
    plt.ylabel("Distance", fontsize=12)
    plt.tight_layout()
    plt.show()
    
    return Z


# ============================================================================
# CLUSTER INTERPRETATION
# ============================================================================

def get_cluster_profile(df, cluster_col, exclude_cols=None):
    """
    Calculate mean values per cluster.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    profile = df.groupby(cluster_col).mean(numeric_only=True)
    profile = profile.drop(columns=[c for c in exclude_cols if c in profile.columns], errors='ignore')
    
    return profile


def get_cluster_profile_vs_overall(df, cluster_col, exclude_cols=None):
    """
    Calculate cluster profile relative to overall mean.
    Values > 1 indicate above average, < 1 indicate below average.
    """
    profile = get_cluster_profile(df, cluster_col, exclude_cols)
    overall_mean = df.drop(columns=[cluster_col], errors='ignore').mean(numeric_only=True)
    
    common_cols = [col for col in profile.columns if col in overall_mean.index]
    profile_aligned = profile[common_cols]
    overall_aligned = overall_mean[common_cols]
    
    profile_ratio = profile_aligned.div(overall_aligned, axis=1)
    
    return profile_ratio


def print_cluster_insights(profile_ratio, top_n=5):
    """
    Print insights for each cluster.
    """
    for cluster in profile_ratio.index:
        print(f"\n{'='*60}")
        print(f"CLUSTER {cluster}")
        print('='*60)
        
        sorted_features = profile_ratio.loc[cluster].sort_values(ascending=False)
        
        print(f"\n>>> Above average features (top {top_n}):")
        for feature, ratio in sorted_features.head(top_n).items():
            pct_above = (ratio - 1) * 100
            feature_clean = feature.replace('_', ' ').title()
            print(f"    {feature_clean}: {pct_above:.1f}% above average")
        
        print(f"\n>>> Below average features (top {top_n}):")
        for feature, ratio in sorted_features.tail(top_n).items():
            pct_below = (1 - ratio) * 100
            feature_clean = feature.replace('_', ' ').title()
            print(f"    {feature_clean}: {pct_below:.1f}% below average")


# ============================================================================
# OUTLIER ASSIGNMENT
# ============================================================================

def assign_outliers_to_clusters(outlier_df, centroids, exclude_cols=None):
    """
    Assign outlier customers to the nearest cluster centroid.
    """
    if exclude_cols is None:
        exclude_cols = []
    
    feature_cols = [col for col in outlier_df.columns if col not in exclude_cols]
    X_outliers = outlier_df[feature_cols].values
    
    closest_idx, closest_dist = pairwise_distances_argmin_min(
        X=X_outliers,
        Y=centroids,
        metric='euclidean'
    )
    
    return closest_idx, closest_dist


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def export_clusters(df, cluster_col, output_path, customer_id_col='customer_id'):
    """
    Export customer clusters to CSV.
    """
    if customer_id_col in df.columns:
        output = df[[customer_id_col, cluster_col]].copy()
    else:
        output = df[[cluster_col]].copy()
        output.index.name = customer_id_col
        output = output.reset_index()
    
    output.to_csv(output_path, index=False)
    print(f"Clusters exported to: {output_path}")
    print(f"Total customers: {len(output)}")
    
    return output