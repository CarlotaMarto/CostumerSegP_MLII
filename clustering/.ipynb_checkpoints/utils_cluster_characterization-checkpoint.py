"""Utilities for customer segment characterization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


CLUSTER_NAMES = {
    0: "Value Seekers",
    1: "Wellness",
    2: "Traditional Shoppers",
    3: "Vegetarians",
    4: "Promotion Hunters",
    5: "Tech Enthusiasts",
    6: "Large Families",
    7: "Loyal Explorers",
}


def load_characterization_data(data_dir="../datasets"):
    """Load features and cluster assignments, including outliers."""
    data_dir = str(data_dir)
    regular = pd.read_csv(f"{data_dir}/info_clustering_unscaled.csv")
    outliers = pd.read_csv(f"{data_dir}/outlier_dataset.csv")
    segments = pd.read_csv(f"{data_dir}/customer_segments.csv")

    features = pd.concat([regular, outliers], ignore_index=True)
    df = features.merge(segments, on="customer_id", how="inner")
    df["cluster_name"] = df["cluster"].map(CLUSTER_NAMES).fillna(df["cluster"].astype(str))
    return df


def cluster_sizes(df, cluster_col="cluster", name_col="cluster_name"):
    """Customers and share by cluster."""
    out = df.groupby([cluster_col, name_col]).size().reset_index(name="customers")
    out["share_%"] = (out["customers"] / out["customers"].sum() * 100).round(1)
    return out.sort_values(cluster_col)


def profile_table(df, features, cluster_col="cluster"):
    """Mean profile by cluster with an OVERALL row."""
    features = [f for f in features if f in df.columns]
    prof = df.groupby(cluster_col)[features].mean()
    prof.loc["OVERALL"] = df[features].mean()
    return prof.round(2)


def scaled_profile(profile_df):
    """Min-max scale each feature across clusters for visual comparison."""
    data = profile_df.drop(index="OVERALL", errors="ignore").astype(float)
    scaled = pd.DataFrame(
        MinMaxScaler().fit_transform(data),
        index=data.index,
        columns=data.columns,
    )
    return scaled.round(2)


def plot_cluster_sizes(size_df):
    """Bar chart of customers per segment."""
    plt.figure(figsize=(9, 4))
    sns.barplot(data=size_df, x="cluster", y="customers", color="#1B4F72")
    plt.title("Customers per segment")
    plt.xlabel("Cluster")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.show()


def plot_profile_heatmap(profile_df, title="Cluster profile"):
    """Heatmap of profile means in original units."""
    data = profile_df.drop(index="OVERALL", errors="ignore")
    plt.figure(figsize=(max(9, data.shape[1] * 0.9), max(4, data.shape[0] * 0.7)))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="Blues", linewidths=0.5)
    plt.title(title)
    plt.ylabel("Cluster")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def plot_scaled_profile(profile_df, title="Normalised segment comparison"):
    """Dot plot of min-max-scaled cluster means."""
    scaled = scaled_profile(profile_df)
    long = scaled.reset_index(names="cluster").melt(
        id_vars="cluster", var_name="feature", value_name="scaled_mean"
    )
    plt.figure(figsize=(10, max(5, scaled.shape[1] * 0.38)))
    sns.scatterplot(data=long, x="scaled_mean", y="feature", hue="cluster", s=90)
    plt.title(title)
    plt.xlabel("Scaled mean within feature")
    plt.ylabel("Feature")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return scaled


def plot_feature_bars(df, features, cluster_col="cluster", max_cols=3):
    """Bar plots of feature averages by cluster."""
    features = [f for f in features if f in df.columns]
    n_cols = min(max_cols, len(features))
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        means = df.groupby(cluster_col)[feature].mean().reset_index()
        sns.barplot(data=means, x=cluster_col, y=feature, ax=ax, color="#7FB3D5")
        ax.set_title(feature)
        ax.set_xlabel("Cluster")

    for ax in axes[len(features):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_boxplot_grid(df, features, cluster_col="cluster", max_cols=3):
    """Boxplots of selected variables by cluster."""
    features = [f for f in features if f in df.columns]
    n_cols = min(max_cols, len(features))
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        sns.boxplot(data=df, x=cluster_col, y=feature, ax=ax, color="#A9CCE3", fliersize=1.5)
        ax.set_title(feature)
        ax.set_xlabel("Cluster")

    for ax in axes[len(features):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def top_deviations(profile_df, n=5):
    """Largest absolute deviations from the overall average by cluster."""
    overall = profile_df.loc["OVERALL"]
    clusters = profile_df.drop(index="OVERALL", errors="ignore")
    rel = (clusters - overall) / overall.replace(0, np.nan)
    rows = []
    for cluster, values in rel.iterrows():
        for feature, value in values.abs().sort_values(ascending=False).head(n).items():
            rows.append({
                "cluster": cluster,
                "feature": feature,
                "relative_difference_%": round(rel.loc[cluster, feature] * 100, 1),
                "cluster_mean": profile_df.loc[cluster, feature],
                "overall_mean": overall[feature],
            })
    return pd.DataFrame(rows)


def export_id_cluster(df, output_path="../datasets/id_and_cluster.csv"):
    """Export customer_id, cluster and cluster_name."""
    cols = ["customer_id", "cluster", "cluster_name"]
    out = df[cols].drop_duplicates("customer_id").sort_values("customer_id")
    out.to_csv(output_path, index=False)
    return out
