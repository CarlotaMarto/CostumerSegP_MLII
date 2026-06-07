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
    "#B87540",  # 0 Vegetarians
    "#B2543D",  # 1 Regulars
    "#7E6A43",  # 2 Wellness
    "#78969B",  # 3 Promoters
    "#D08F78",  # 4 Loyalists
    "#2F7A6A",  # 5 Families
    "#5A3516",  # 6 Economizers
    "#9B7DB8",  # 7 Techies
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


CLUSTER_NAMES = {
    0: "Vegetarians",   # highest vegetable spend, lowest meat
    1: "Regulars",      # moderate across all features
    2: "Wellness",      # highest hygiene spend, high vegetables
    3: "Promoters",     # highest promotion-purchase rate
    4: "Loyalists",     # highest loyalty flag, highest groceries spend
    5: "Families",      # highest children count, highest meat spend
    6: "Economizers",   # fewest store visits, moderate spend
    7: "Techies",       # highest electronics and videogame spend
}

def load_characterization_data(data_dir="../datasets"):
    """Load features and cluster assignments, including outliers."""
    data_dir = str(data_dir)
    regular = pd.read_csv(f"{data_dir}/info_clustering_unscaled.csv")
    outliers = pd.read_csv(f"{data_dir}/outlier_dataset.csv")
    segments = pd.read_csv(f"{data_dir}/customer_segments.csv")

    features = pd.concat([regular, outliers], ignore_index=True)
    df = features.merge(segments, on="customer_id", how="inner")

    original = pd.read_csv(f"{data_dir}/customer_info.csv")
    missing_granular = [c for c in ["lifetime_spend_electronics", "lifetime_spend_videogames"]
                        if c not in df.columns and c in original.columns]
    if missing_granular:
        df = df.merge(original[["customer_id"] + missing_granular], on="customer_id", how="left")

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



def _radar_labels(features):
    """Shorten feature names for radar chart axes."""
    subs = {
        "lifetime_spend_": "",
        "annual_spend_": "",
        "percentage_of_products_bought_promotion": "promotion_%",
        "log_total_spend": "log_spend",
        "lifetime_total_distinct_products": "distinct_products",
        "distinct_stores_visited": "stores_visited",
    }
    out = []
    for f in features:
        label = f
        for old, new in subs.items():
            label = label.replace(old, new)
        out.append(label)
    return out


def plot_radar_profiles(
    profile_df,
    features=None,
    cluster_names=None,
    title="Segment radar profiles",
    max_cols=4,
):
    """One radar chart per cluster (small multiples, min-max scaled)."""
    if cluster_names is None:
        cluster_names = CLUSTER_NAMES

    data = scaled_profile(profile_df)
    if features is None:
        features = list(data.columns)
    features = [f for f in features if f in data.columns]
    if not features:
        return pd.DataFrame()

    plot_data = data[features]
    labels = _radar_labels(features)
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]

    n_clusters = len(plot_data)
    n_cols = min(max_cols, n_clusters)
    n_rows = int(np.ceil(n_clusters / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 4.2 * n_rows),
        subplot_kw={"projection": "polar"},
    )
    axes = np.atleast_1d(axes).ravel()

    for pos, (cluster, row) in enumerate(plot_data.iterrows()):
        ax = axes[pos]
        values = row.tolist() + row.tolist()[:1]
        color = CLUSTER_PALETTE[pos % len(CLUSTER_PALETTE)]
        label = cluster_names.get(cluster, f"Cluster {cluster}") if cluster_names else f"Cluster {cluster}"
        ax.plot(angles, values, color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.25)
        ax.set_title(label, pad=12, fontsize=11, fontweight="bold")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
        ax.set_ylim(0, 1)

    for ax in axes[n_clusters:]:
        ax.axis("off")

    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    return plot_data.round(2)


def plot_radar_combined(
    profile_df,
    features=None,
    cluster_names=None,
    title="Segment comparison — radar chart",
    alpha=0.15,
    figsize=(8, 8),
):
    """All clusters overlaid on a single radar chart for direct comparison."""
    if cluster_names is None:
        cluster_names = CLUSTER_NAMES

    data = scaled_profile(profile_df)
    if features is None:
        features = list(data.columns)
    features = [f for f in features if f in data.columns]
    if not features:
        return pd.DataFrame()

    plot_data = data[features]
    labels = _radar_labels(features)
    n = len(features)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=8, color="grey")
    ax.set_ylim(0, 1)
    ax.set_title(title, pad=20, fontsize=13, fontweight="bold")

    for pos, (cluster, row) in enumerate(plot_data.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        color = CLUSTER_PALETTE[pos % len(CLUSTER_PALETTE)]
        name = cluster_names.get(cluster, f"Cluster {cluster}") if cluster_names else f"Cluster {cluster}"
        ax.plot(angles, values, color=color, linewidth=2, label=name)
        ax.fill(angles, values, color=color, alpha=alpha)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.3, 1.15),
        fontsize=9,
        title="Segment",
        title_fontsize=9,
    )
    plt.tight_layout()
    plt.show()
    return plot_data.round(2)


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
    cols = [c for c in df.columns if c.startswith("lifetime_spend_")]
    if {"lifetime_spend_electronics", "lifetime_spend_videogames"}.issubset(cols):
        cols = [c for c in cols if c != "lifetime_spend_technology"]
    return cols


def behavioural_profile_columns(df):
    """Main behavioural and demographic variables used for interpretation."""
    return [
        "log_total_spend",
        "percentage_of_products_bought_promotion",
        "distinct_stores_visited",
        "lifetime_total_distinct_products",
        "tenure",
        "total_children",
        "number_complaints",
        "customer_loyalty_flag",
    ]


def key_plot_columns(df):
    """Compact feature list for barplots and boxplots."""
    return [
        "lifetime_spend_groceries",
        "lifetime_spend_vegetables",
        "lifetime_spend_alcohol_drinks",
        "lifetime_spend_meat",
        "lifetime_spend_fish",
        "lifetime_spend_hygiene",
        "lifetime_spend_petfood",
        "lifetime_spend_electronics",
        "lifetime_spend_videogames",
        "log_total_spend",
        "percentage_of_products_bought_promotion",
        "total_children",
        "number_complaints",
    ]


def binary_profile_columns(df):
    """Binary attributes used for simple segment checks."""
    return ["customer_loyalty_flag"]


def household_profile_columns(df):
    """Household and complaints variables used for simple segment checks."""
    return ["total_children", "number_complaints"]


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


def plot_cluster_summary_card(
    df,
    cluster_id,
    profile_df,
    cluster_col="cluster",
    cluster_names=None,
    spend_features=None,
    stat_features=None,
    figsize=(14, 5),
):
    """Three-panel summary card for one cluster: spend radar, key differentiators, stats table."""
    if cluster_names is None:
        cluster_names = CLUSTER_NAMES

    segment_name = cluster_names.get(cluster_id, f"Cluster {cluster_id}")
    color = CLUSTER_PALETTE[int(cluster_id) % len(CLUSTER_PALETTE)]

    if spend_features is None:
        spend_features = [c for c in df.columns if c.startswith("annual_spend_")]
        if not spend_features:
            spend_features = [c for c in df.columns if c.startswith("lifetime_spend_")]

    if stat_features is None:
        stat_features = [
            "total_children", "tenure", "log_total_spend",
            "percentage_of_products_bought_promotion", "distinct_stores_visited",
            "number_complaints", "customer_loyalty_flag",
        ]

    cluster_df = df[df[cluster_col] == cluster_id]
    n_customers = len(cluster_df)
    pct = n_customers / len(df) * 100

    fig = plt.figure(figsize=figsize)
    fig.suptitle(
        f"Cluster {cluster_id}  ·  {segment_name}  "
        f"({n_customers:,} customers, {pct:.1f}%)",
        fontsize=13, fontweight="bold", x=0.5, y=1.01,
    )

    gs = fig.add_gridspec(1, 3, wspace=0.45)

    ax_radar = fig.add_subplot(gs[0, 0], projection="polar")
    spend_prof = profile_df.drop(index="OVERALL", errors="ignore")
    avail_spend = [f for f in spend_features if f in spend_prof.columns]
    if avail_spend:
        scaled = pd.DataFrame(
            MinMaxScaler().fit_transform(spend_prof[avail_spend].astype(float)),
            index=spend_prof.index,
            columns=avail_spend,
        )
        if cluster_id in scaled.index:
            vals = scaled.loc[cluster_id, avail_spend].tolist()
            labels = _radar_labels(avail_spend)
            angles = np.linspace(0, 2 * np.pi, len(avail_spend), endpoint=False).tolist()
            angles += angles[:1]
            vals += vals[:1]
            ax_radar.plot(angles, vals, color=color, linewidth=2)
            ax_radar.fill(angles, vals, color=color, alpha=0.25)
            ax_radar.set_xticks(angles[:-1])
            ax_radar.set_xticklabels(labels, fontsize=7)
            ax_radar.set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_radar.set_yticklabels(["", "", "", ""], fontsize=6)
            ax_radar.set_ylim(0, 1)
    ax_radar.set_title("Spend profile\n(scaled)", fontsize=9, pad=10)

    ax_bar = fig.add_subplot(gs[0, 1])
    if "OVERALL" in profile_df.index and cluster_id in profile_df.index:
        overall = profile_df.loc["OVERALL"]
        cluster_vals = profile_df.loc[cluster_id]
        rel = ((cluster_vals - overall) / overall.replace(0, np.nan)).dropna()
        rel = rel[rel.index.isin(stat_features)]
        top_pos = rel.nlargest(5)
        top_neg = rel.nsmallest(3)
        top = pd.concat([top_pos, top_neg]).sort_values()
        bar_colors = [ACCENT_COLOR if v < 0 else color for v in top.values]
        short_labels = _radar_labels(top.index.tolist())
        ax_bar.barh(short_labels, top.values * 100, color=bar_colors)
        ax_bar.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax_bar.set_xlabel("Deviation from overall mean (%)", fontsize=8)
        ax_bar.tick_params(axis="y", labelsize=8)
        ax_bar.set_title("Key differentiators", fontsize=9)
    else:
        ax_bar.axis("off")

    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis("off")

    stat_labels = {
        "total_children": "Avg children",
        "tenure": "Avg tenure (yrs)",
        "log_total_spend": "Avg log-spend",
        "percentage_of_products_bought_promotion": "Promo purchase %",
        "distinct_stores_visited": "Avg stores visited",
        "number_complaints": "Avg complaints",
        "customer_loyalty_flag": "Loyalty card %",
    }

    lines = [f"{'Metric':<28}{'Value':>8}", "-" * 38]
    for feat in stat_features:
        if feat not in cluster_df.columns:
            continue
        val = cluster_df[feat].mean()
        label = stat_labels.get(feat, feat)
        if feat == "customer_loyalty_flag":
            lines.append(f"{label:<28}{val * 100:>7.1f}%")
        elif feat == "percentage_of_products_bought_promotion":
            lines.append(f"{label:<28}{val * 100:>7.1f}%")
        else:
            lines.append(f"{label:<28}{val:>8.2f}")

    ax_txt.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax_txt.transAxes,
        fontsize=8, verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor=LIGHT_COLOR, alpha=0.7),
    )
    ax_txt.set_title("Segment statistics", fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_all_cluster_cards(
    df,
    profile_df,
    cluster_col="cluster",
    cluster_names=None,
    **kwargs,
):
    """Plot one summary card for every cluster in sorted order."""
    if cluster_names is None:
        cluster_names = CLUSTER_NAMES
    for cid in sorted(df[cluster_col].unique()):
        plot_cluster_summary_card(
            df, cid, profile_df,
            cluster_col=cluster_col,
            cluster_names=cluster_names,
            **kwargs,
        )

