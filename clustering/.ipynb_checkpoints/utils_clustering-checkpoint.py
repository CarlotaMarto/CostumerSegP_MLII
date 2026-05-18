"""
Utility functions for the Customer Segmentation clustering notebook.

The functions in this file keep the notebook cleaner and make the clustering
workflow easier to reuse and explain.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min


def evaluate_clustering(X, labels, model_name):
    """
    Evaluates a clustering solution using Silhouette Score.
    If the model produces fewer than two valid clusters, the score is returned as NaN.
    """
    labels = np.array(labels)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    if n_clusters < 2:
        return pd.DataFrame({
            "Model": [model_name],
            "Clusters": [n_clusters],
            "Silhouette Score": [np.nan]
        })

    return pd.DataFrame({
        "Model": [model_name],
        "Clusters": [n_clusters],
        "Silhouette Score": [silhouette_score(X, labels)]
    })


def cluster_distribution(labels):
    """
    Returns the number of observations assigned to each cluster.
    """
    return pd.Series(labels).value_counts().sort_index()


def create_cluster_profile(df_clustered, cluster_col, id_col="customer_id"):
    """
    Creates a cluster profile with average feature values by cluster.
    Identifier columns are excluded from the profile.
    """
    df_profile = df_clustered.drop(columns=[id_col], errors="ignore")

    profile = df_profile.groupby(cluster_col).mean(numeric_only=True)

    cluster_size = df_clustered[cluster_col].value_counts().sort_index()
    cluster_percentage = np.round(cluster_size / len(df_clustered) * 100, 2)

    profile.insert(0, "cluster_size", cluster_size)
    profile.insert(1, "cluster_percentage", cluster_percentage)

    return profile.round(2)


def plot_elbow_and_silhouette(kmeans_results):
    """
    Plots the Elbow Method and Silhouette Scores for K-Means.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(kmeans_results["k"], kmeans_results["inertia"], marker="o")
    axes[0].set_title("K-Means Elbow Method")
    axes[0].set_xlabel("Number of Clusters")
    axes[0].set_ylabel("Inertia")
    axes[0].grid(True)

    axes[1].plot(kmeans_results["k"], kmeans_results["silhouette_score"], marker="o")
    axes[1].set_title("K-Means Silhouette Scores")
    axes[1].set_xlabel("Number of Clusters")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_pca_clusters(X, labels, title):
    """
    Visualizes clustering results using the first two PCA components.
    """
    pca = PCA(n_components=2)
    embedding = pca.fit_transform(X)

    print(f"Explained variance with 2 components: {pca.explained_variance_ratio_.sum():.4f}")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=embedding[:, 0],
        y=embedding[:, 1],
        hue=labels,
        palette="Set2",
        s=40
    )

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(title="Cluster")
    plt.show()

    return embedding, pca


def plot_silhouette_analysis(X, labels, title="Silhouette Plot"):
    """
    Creates a silhouette plot for the selected clustering solution.
    """
    labels = np.array(labels)
    unique_clusters = sorted([cluster for cluster in set(labels) if cluster != -1])

    if len(unique_clusters) < 2:
        print("Silhouette plot requires at least 2 valid clusters.")
        return

    silhouette_avg = silhouette_score(X, labels)
    sample_silhouette_values = silhouette_samples(X, labels)

    plt.figure(figsize=(10, 6))

    y_lower = 10
    for cluster in unique_clusters:
        cluster_values = sample_silhouette_values[labels == cluster]
        cluster_values.sort()

        size_cluster = cluster_values.shape[0]
        y_upper = y_lower + size_cluster

        plt.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            cluster_values,
            alpha=0.7
        )

        plt.text(-0.05, y_lower + 0.5 * size_cluster, str(cluster))
        y_lower = y_upper + 10

    plt.axvline(
        x=silhouette_avg,
        color="red",
        linestyle="--",
        label=f"Average = {silhouette_avg:.3f}"
    )

    plt.title(title)
    plt.xlabel("Silhouette coefficient")
    plt.ylabel("Cluster")
    plt.legend()
    plt.show()


def plot_cluster_sizes(df, cluster_col):
    """
    Plots the number of customers in each cluster.
    """
    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=df,
        x=cluster_col,
        order=sorted(df[cluster_col].unique())
    )
    plt.title("Customer Distribution by Cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Customers")
    plt.show()


def plot_cluster_feature_bars(df, cluster_col, features):
    """
    Plots average feature values by cluster for selected variables.
    """
    for col in features:
        if col in df.columns:
            plt.figure(figsize=(8, 4))
            sns.barplot(
                data=df,
                x=cluster_col,
                y=col,
                errorbar=None
            )
            plt.title(f"Average {col} by Cluster")
            plt.xlabel("Cluster")
            plt.ylabel(col)
            plt.show()
        else:
            print(f"Skipped missing feature: {col}")


def get_top_cluster_features(profile, top_n=5):
    """
    Shows the highest and lowest standardized characteristics per cluster.
    """
    profile_features = profile.drop(
        columns=["cluster_size", "cluster_percentage"],
        errors="ignore"
    )

    rows = []

    for cluster in profile_features.index:
        sorted_values = profile_features.loc[cluster].sort_values(ascending=False)

        rows.append({
            "cluster": cluster,
            "top_positive_features": list(sorted_values.head(top_n).index),
            "top_negative_features": list(sorted_values.tail(top_n).index)
        })

    return pd.DataFrame(rows)


def plot_centroid_comparison(centroids_df, cluster_col="cluster"):
    """
    Plots a compact comparison of cluster centroids using MinMax scaling.
    This is only used for visualization.
    """
    plot_df = centroids_df.copy()

    if cluster_col not in plot_df.columns:
        plot_df.insert(0, cluster_col, plot_df.index)

    features = plot_df.drop(columns=[cluster_col])

    scaled_features = pd.DataFrame(
        MinMaxScaler().fit_transform(features),
        columns=features.columns,
        index=plot_df[cluster_col]
    )

    long_df = scaled_features.reset_index().melt(
        id_vars=cluster_col,
        var_name="feature",
        value_name="scaled_value"
    )

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=long_df,
        x="scaled_value",
        y="feature",
        hue=cluster_col,
        s=90,
        palette="tab10"
    )
    plt.title("Scaled Centroid Comparison by Cluster")
    plt.xlabel("Scaled centroid value")
    plt.ylabel("Feature")
    plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()


def add_segment_names(df, cluster_col="kmeans_cluster", name_col="segment_name"):
    """
    Adds final business segment names based on the interpreted K-Means cluster profile.
    """
    segment_names = {
        0: "Promotion-Oriented Regulars",
        1: "Healthy Lifestyle Buyers",
        2: "Premium Grocery & Pet Buyers",
        3: "Tech-Focused Customers",
        4: "Family Loyal Customers"
    }

    df_named = df.copy()
    df_named[name_col] = df_named[cluster_col].map(segment_names)

    return df_named, segment_names


def add_recommendations(df, name_col="segment_name", action_col="recommended_action"):
    """
    Adds simple business recommendations for each interpreted segment.
    """
    recommendations = {
        "Promotion-Oriented Regulars": "Use discount campaigns, personalized coupons, and promotion bundles.",
        "Healthy Lifestyle Buyers": "Promote fresh produce, hygiene products, wellness bundles, and healthy lifestyle offers.",
        "Premium Grocery & Pet Buyers": "Use cross-category bundles involving meat, fish, pet food, and premium grocery items.",
        "Tech-Focused Customers": "Target with technology offers, new product launches, and digital engagement campaigns.",
        "Family Loyal Customers": "Offer family packs, loyalty rewards, bulk discounts, and household-oriented promotions."
    }

    df_out = df.copy()
    df_out[action_col] = df_out[name_col].map(recommendations)
    return df_out


def assign_outliers_to_nearest_centroid(outlier_df, feature_columns, kmeans_model, cluster_col="kmeans_cluster"):
    """
    Assigns already-preprocessed outliers to the nearest K-Means centroid.

    Important: the outlier dataframe must be in the same feature space and scaling
    as the dataset used to train K-Means.
    """
    missing_cols = [col for col in feature_columns if col not in outlier_df.columns]

    if missing_cols:
        raise ValueError(
            "Outlier dataset is missing required clustering columns: "
            + ", ".join(missing_cols)
        )

    X_outliers = outlier_df[feature_columns]
    labels, distances = pairwise_distances_argmin_min(
        X_outliers,
        kmeans_model.cluster_centers_
    )

    assigned = outlier_df.copy()
    assigned[cluster_col] = labels
    assigned["distance_to_centroid"] = distances

    return assigned
