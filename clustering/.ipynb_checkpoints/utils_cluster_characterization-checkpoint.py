"""Utilities for customer segment characterization."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from IPython.display import display
from sklearn.preprocessing import MinMaxScaler

PROJECT_PALETTE = [
    "#B87540",
    "#B2543D",
    "#7E6A43",
    "#A8B7BA",
    "#D8C0B4",
    "#C8AB8C",
    "#5A3516",
    "#B98F70",
]
CLUSTER_PALETTE = [
    "#B87540",  # 0 Sale-Oriented Shoppers
    "#B2543D",  # 1 Early Bird Pet Owners
    "#7E6A43",  # 2 Gamers & Technophiles
    "#78969B",  # 3 Vegetarians & Vegans
    "#D08F78",  # 4 Bargain Hunters
    "#2F7A6A",  # 5 Loyalists
    "#5A3516",  # 6 Health & Pet Shoppers
    "#9B7DB8",  # 7 Regular Shoppers
    "#A8B7BA",  # 8 Large Families
]
MAIN_COLOR = "#B87540"
ACCENT_COLOR = "#B2543D"
NOTE_COLOR = "#7E6A43"
SECONDARY_COLOR = "#A8B7BA"
LIGHT_COLOR = "#F3EEE6"
DARK_COLOR = "#5A3516"


def sequential_cmap():
    return LinearSegmentedColormap.from_list(
        "project_sequential",
        [LIGHT_COLOR, "#C8AB8C", MAIN_COLOR, ACCENT_COLOR, DARK_COLOR],
    )


def diverging_cmap():
    return LinearSegmentedColormap.from_list(
        "project_diverging",
        [ACCENT_COLOR, LIGHT_COLOR, SECONDARY_COLOR],
    )


def cluster_cmap():
    return ListedColormap(CLUSTER_PALETTE)


CLUSTER_NAMES = {
    0: "Sale-Oriented Shoppers",
    1: "Early Bird Pet Owners",
    2: "Gamers & Technophiles",
    3: "Vegetarians & Vegans",
    4: "Bargain Hunters",
    5: "Loyalists",
    6: "Health & Pet Shoppers",
    7: "Regular Shoppers",
    8: "Large Families",
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
    sns.barplot(data=size_df, x="cluster", y="customers", color=MAIN_COLOR)
    plt.title("Customers per segment")
    plt.xlabel("Cluster")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.show()


def plot_profile_heatmap(profile_df, title="Cluster profile"):
    """Heatmap of profile means in original units."""
    data = profile_df.drop(index="OVERALL", errors="ignore")
    plt.figure(figsize=(max(9, data.shape[1] * 0.9), max(4, data.shape[0] * 0.7)))
    sns.heatmap(data, annot=True, fmt=".1f", cmap=sequential_cmap(), linewidths=0.5)
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
    sns.scatterplot(data=long, x="scaled_mean", y="feature", hue="cluster", palette=CLUSTER_PALETTE, s=90)
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
        sns.barplot(data=means, x=cluster_col, y=feature, ax=ax, color=SECONDARY_COLOR)
        ax.set_title(feature)
        ax.set_xlabel("Cluster")

    for ax in axes[len(features):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()



def plot_binary_share_by_cluster(df, columns, cluster_col="cluster"):
    """Plot the percentage of customers with selected binary attributes by cluster."""
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return pd.DataFrame()

    rows = []
    for col in columns:
        means = df.groupby(cluster_col)[col].mean().reset_index(name="share")
        means["feature"] = col
        rows.append(means)
    summary = pd.concat(rows, ignore_index=True)
    summary["share_%"] = summary["share"] * 100

    plt.figure(figsize=(10, 4.5))
    sns.barplot(data=summary, x=cluster_col, y="share_%", hue="feature", palette=[MAIN_COLOR, SECONDARY_COLOR, NOTE_COLOR])
    plt.title("Binary profile by cluster")
    plt.xlabel("Cluster")
    plt.ylabel("Percentage of customers")
    plt.legend(title="Feature", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
    return summary[[cluster_col, "feature", "share_%"]].round(1)


def plot_mean_profile_bars(df, columns, cluster_col="cluster", title="Mean profile by cluster"):
    """Plot mean values of selected numeric attributes by cluster."""
    columns = [c for c in columns if c in df.columns]
    if not columns:
        return pd.DataFrame()

    summary = df.groupby(cluster_col)[columns].mean().reset_index()
    long = summary.melt(id_vars=cluster_col, var_name="feature", value_name="mean")

    n_cols = min(2, len(columns))
    n_rows = int(np.ceil(len(columns) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, columns):
        sub = long[long["feature"] == feature]
        sns.barplot(data=sub, x=cluster_col, y="mean", ax=ax, color=MAIN_COLOR)
        ax.set_title(feature)
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Mean")

    for ax in axes[len(columns):]:
        ax.axis("off")
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()
    return summary.round(2)

def plot_boxplot_grid(df, features, cluster_col="cluster", max_cols=3):
    """Boxplots of selected variables by cluster."""
    features = [f for f in features if f in df.columns]
    n_cols = min(max_cols, len(features))
    n_rows = int(np.ceil(len(features) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.2 * n_cols, 3.8 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, feature in zip(axes, features):
        sns.boxplot(data=df, x=cluster_col, y=feature, ax=ax, color=SECONDARY_COLOR, fliersize=1.5)
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
    """Export the final customer segment assignment."""
    cols = ["customer_id", "cluster"]
    if "cluster_name" in df.columns:
        cols.append("cluster_name")
    out = df[cols].drop_duplicates("customer_id").sort_values("customer_id")
    out.to_csv(output_path, index=False)
    return out



def spend_columns(df):
    """Lifetime spend variables available in the characterization data."""
    return [c for c in df.columns if c.startswith("lifetime_spend_")]


def behavioural_profile_columns(df):
    """Main behavioural and demographic variables used for interpretation."""
    return [c for c in [
        "log_total_spend",
        "percentage_of_products_bought_promotion",
        "distinct_stores_visited",
        "lifetime_total_distinct_products",
        "customer_age",
        "education_level",
        "tenure",
        "total_children",
        "number_complaints",
        "customer_loyalty_flag",
        "is_male",
    ] if c in df.columns]


def key_plot_columns(df):
    """Compact feature list for barplots and boxplots."""
    return [c for c in [
        "lifetime_spend_groceries",
        "lifetime_spend_vegetables",
        "lifetime_spend_alcohol_drinks",
        "lifetime_spend_meat",
        "lifetime_spend_fish",
        "lifetime_spend_hygiene",
        "lifetime_spend_petfood",
        "lifetime_spend_technology",
        "log_total_spend",
        "percentage_of_products_bought_promotion",
        "customer_age",
        "total_children",
        "number_complaints",
    ] if c in df.columns]


def binary_profile_columns(df):
    """Binary attributes used for simple segment checks."""
    return [c for c in ["customer_loyalty_flag", "is_male"] if c in df.columns]


def household_profile_columns(df):
    """Household and complaints variables used for simple segment checks."""
    return [c for c in ["total_children", "number_complaints"] if c in df.columns]


def plot_simple_profile_checks(df):
    """Plot loyalty, gender, household size and complaints by cluster."""
    binary_summary = plot_binary_share_by_cluster(df, binary_profile_columns(df))
    display(binary_summary)

    household_summary = plot_mean_profile_bars(
        df,
        household_profile_columns(df),
        title="Household and complaints profile by cluster",
    )
    display(household_summary)
    return binary_summary, household_summary

