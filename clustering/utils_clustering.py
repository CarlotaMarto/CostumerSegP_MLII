import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA


# ============================================================
# DATA VALIDATION
# ============================================================

def validate_clustering_data(df):
    """
    Prints basic validation checks before clustering.
    """
    print("Dataset shape:", df.shape)
    print("Missing values:", df.isna().sum().sum())
    print("Duplicated rows:", df.duplicated().sum())

    display(df.head())
    display(df.describe().T)


# ============================================================
# SCALING
# ============================================================

def create_scaled_versions(df, include_no_scaling=False):
    """
    Creates scaled versions of the input dataset.

    Parameters:
    - df: input DataFrame
    - include_no_scaling: whether to include the unscaled data as a comparison

    Returns:
    - scaled_versions: dictionary with scaled arrays
    - fitted_scalers: dictionary with fitted scaler objects
    """
    scaled_versions = {}
    fitted_scalers = {}

    if include_no_scaling:
        scaled_versions["no_scaling"] = df.values
        fitted_scalers["no_scaling"] = None

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()
    }

    for name, scaler in scalers.items():
        scaled_versions[name] = scaler.fit_transform(df)
        fitted_scalers[name] = scaler

    return scaled_versions, fitted_scalers


# ============================================================
# K-MEANS EVALUATION
# ============================================================

def evaluate_kmeans(X_scaled, k_range=range(2, 11), random_state=42, n_init=20):
    """
    Evaluates K-Means for different numbers of clusters.
    """
    results = []

    for k in k_range:
        model = KMeans(
            n_clusters=k,
            random_state=random_state,
            n_init=n_init
        )

        labels = model.fit_predict(X_scaled)

        results.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(X_scaled, labels),
            "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            "calinski_harabasz": calinski_harabasz_score(X_scaled, labels)
        })

    return pd.DataFrame(results)


def evaluate_kmeans_for_scalers(scaled_versions, k_range=range(2, 11), random_state=42):
    """
    Runs K-Means evaluation for every scaled version of the dataset.
    """
    results = {}

    for scaler_name, X_scaled in scaled_versions.items():
        print(f"K-Means evaluation with {scaler_name} scaling")
        results[scaler_name] = evaluate_kmeans(
            X_scaled,
            k_range=k_range,
            random_state=random_state
        )
        display(results[scaler_name])

    return results


def plot_kmeans_metrics(kmeans_results):
    """
    Plots elbow curve and silhouette score for each scaling method.
    """
    for scaler_name, results in kmeans_results.items():
        plt.figure(figsize=(8, 5))
        plt.plot(results["k"], results["inertia"], marker="o")
        plt.title(f"Elbow Curve - {scaler_name.capitalize()} Scaling")
        plt.xlabel("Number of clusters")
        plt.ylabel("Inertia")
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.plot(results["k"], results["silhouette"], marker="o")
        plt.title(f"Silhouette Score - {scaler_name.capitalize()} Scaling")
        plt.xlabel("Number of clusters")
        plt.ylabel("Silhouette Score")
        plt.grid(True)
        plt.show()


# ============================================================
# HIERARCHICAL CLUSTERING
# ============================================================

def plot_ward_dendrogram(X_scaled, sample_size=3000, random_state=42, truncate_level=5):
    """
    Plots a Ward linkage dendrogram using a sample of the dataset.
    """
    sample_size = min(sample_size, X_scaled.shape[0])

    sample_indices = np.random.RandomState(random_state).choice(
        X_scaled.shape[0],
        size=sample_size,
        replace=False
    )

    X_sample = X_scaled[sample_indices]
    linked = linkage(X_sample, method="ward")

    plt.figure(figsize=(16, 8))
    dendrogram(linked, truncate_mode="level", p=truncate_level)
    plt.title("Hierarchical Clustering Dendrogram - Ward Linkage")
    plt.xlabel("Customers")
    plt.ylabel("Distance")
    plt.show()

    return linked


def fit_ward_clustering(X_scaled, n_clusters):
    """
    Fits Agglomerative Clustering using Ward linkage.
    """
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage="ward"
    )

    labels = model.fit_predict(X_scaled)
    return model, labels


# ============================================================
# FINAL CLUSTERING SOLUTION
# ============================================================

def fit_kmeans_final(X_scaled, n_clusters, random_state=42, n_init=20):
    """
    Fits the final K-Means model.
    """
    model = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=n_init
    )

    labels = model.fit_predict(X_scaled)
    return model, labels


def compare_cluster_solutions(labels_1, labels_2, name_1="K-Means", name_2="Ward"):
    """
    Compares two clustering solutions using a crosstab.
    """
    comparison = pd.crosstab(
        labels_1,
        labels_2,
        rownames=[name_1],
        colnames=[name_2]
    )

    display(comparison)
    return comparison


def plot_cluster_sizes(df, cluster_col):
    """
    Plots the number of observations in each cluster.
    """
    df[cluster_col].value_counts().sort_index().plot(kind="bar", figsize=(8, 5))

    plt.title(f"Number of Customers per {cluster_col}")
    plt.xlabel("Cluster")
    plt.ylabel("Number of customers")
    plt.grid(axis="y", alpha=0.3)
    plt.show()


# ============================================================
# CLUSTER PROFILING
# ============================================================

def calculate_cluster_means(df, cluster_col):
    """
    Calculates the mean value of each feature by cluster.
    """
    return df.groupby(cluster_col).mean(numeric_only=True)


def calculate_cluster_profile(df, cluster_col, original_features):
    """
    Compares each cluster mean against the global mean.

    Values above 1 indicate the cluster is above the global average.
    Values below 1 indicate the cluster is below the global average.
    """
    cluster_means = df.groupby(cluster_col)[original_features].mean()
    global_mean = df[original_features].mean()

    profile = cluster_means.div(global_mean, axis=1)
    return profile


def plot_cluster_profile_heatmap(cluster_profile):
    """
    Plots a heatmap comparing cluster averages against the global average.
    """
    plt.figure(figsize=(16, 10))

    sns.heatmap(
        cluster_profile.T,
        cmap="coolwarm",
        center=1,
        annot=False
    )

    plt.title("Cluster Profile Compared to Global Mean")
    plt.xlabel("Cluster")
    plt.ylabel("Feature")
    plt.show()


def show_top_cluster_drivers(cluster_profile, cluster_id, top_n=8):
    """
    Shows the features that most distinguish a specific cluster from the global average.
    """
    profile = cluster_profile.loc[cluster_id]

    drivers = pd.DataFrame({
        "feature": profile.index,
        "relative_to_global_mean": profile.values,
        "absolute_distance_from_1": np.abs(profile.values - 1)
    })

    drivers = drivers.sort_values(
        "absolute_distance_from_1",
        ascending=False
    ).head(top_n)

    return drivers


# ============================================================
# DIMENSIONALITY REDUCTION VISUALIZATION
# ============================================================

def plot_pca_clusters(X_scaled, labels, title="Customer Segments - PCA Projection"):
    """
    Uses PCA to project the clustering solution into two dimensions.
    """
    pca = PCA(n_components=2, random_state=42)
    embedding = pca.fit_transform(X_scaled)

    pca_df = pd.DataFrame(
        embedding,
        columns=["PC1", "PC2"]
    )

    pca_df["cluster"] = labels

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=pca_df,
        x="PC1",
        y="PC2",
        hue="cluster",
        palette="tab10",
        s=20,
        alpha=0.6
    )

    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster")
    plt.show()

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance by PC1: {explained_variance[0]:.2%}")
    print(f"Explained variance by PC2: {explained_variance[1]:.2%}")

    return pca_df, pca


def plot_umap_clusters(X_scaled, labels, title="Customer Segments - UMAP Projection"):
    """
    Uses UMAP to project the clustering solution into two dimensions.
    Requires umap-learn to be installed.
    """
    try:
        import umap
    except ImportError:
        raise ImportError("UMAP is not installed. Install it with: pip install umap-learn")

    umap_model = umap.UMAP(
        n_neighbors=30,
        min_dist=0.15,
        random_state=42
    )

    embedding = umap_model.fit_transform(X_scaled)

    umap_df = pd.DataFrame(
        embedding,
        columns=["UMAP1", "UMAP2"]
    )

    umap_df["cluster"] = labels

    plt.figure(figsize=(12, 8))

    sns.scatterplot(
        data=umap_df,
        x="UMAP1",
        y="UMAP2",
        hue="cluster",
        palette="tab10",
        s=20,
        alpha=0.6
    )

    plt.title(title)
    plt.xlabel("UMAP1")
    plt.ylabel("UMAP2")
    plt.legend(title="Cluster")
    plt.show()

    return umap_df, umap_model


# ============================================================
# EXPORT
# ============================================================

def export_clustered_dataset(df, output_path):
    """
    Saves the clustered dataset to CSV.
    """
    df.to_csv(output_path, index=True)

    print(f"Clustered dataset saved to: {output_path}")
    print("Final shape:", df.shape)
