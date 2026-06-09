from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import utils_cluster_characterization as ucc

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", None)

DATA_DIR = Path("../datasets")
df = ucc.load_characterization_data(DATA_DIR)
print("Characterization dataset shape:", df.shape)
df.head()

PROJECT_PALETTE = ['#B87540', '#B2543D', '#7E6A43', '#A8B7BA', '#D8C0B4', '#C8AB8C', '#5A3516', '#B98F70']
sns.set_theme(style='whitegrid', palette=PROJECT_PALETTE)
size_profile = ucc.cluster_sizes(df)
display(size_profile)
ucc.plot_cluster_sizes(size_profile)

spend_cols = ucc.spend_columns(df)
spend_profile = ucc.profile_table(df, spend_cols)
display(spend_profile)
ucc.plot_profile_heatmap(spend_profile, "Spend profile by segment")
profile_cols = ucc.behavioural_profile_columns(df)
behaviour_profile = ucc.profile_table(df, profile_cols)
display(behaviour_profile)
ucc.plot_profile_heatmap(behaviour_profile, "Behavioural and demographic profile")
binary_summary, household_summary = ucc.plot_simple_profile_checks(df)
combined_profile = ucc.profile_table(df, spend_cols + profile_cols)
scaled_profile = ucc.plot_scaled_profile(combined_profile)
display(scaled_profile)
radar_features = [
    "lifetime_spend_groceries",
    "lifetime_spend_vegetables",
    "lifetime_spend_hygiene",
    "lifetime_spend_electronics",
    "lifetime_spend_videogames",
    "percentage_of_products_bought_promotion",
    "lifetime_spend_meat",
    "total_children",
    "customer_loyalty_flag",
]

radar_profile = ucc.plot_radar_profiles(
    combined_profile,
    features=radar_features,
    cluster_names=ucc.CLUSTER_NAMES,
    title="Segment radar profiles",
)
display(radar_profile)

ucc.plot_radar_combined(
    combined_profile,
    features=radar_features,
    cluster_names=ucc.CLUSTER_NAMES,
    title="Segment radar ù all clusters overlaid",
)
key_plot_cols = ucc.key_plot_columns(df)
ucc.plot_feature_bars(df, key_plot_cols)
ucc.plot_boxplot_grid(df, key_plot_cols)
deviations = ucc.top_deviations(combined_profile, n=6)
display(deviations)

ucc.plot_all_cluster_cards(
    df,
    combined_profile,
    cluster_col='cluster',
    cluster_names=ucc.CLUSTER_NAMES,
)
if {"latitude", "longitude"}.issubset(df.columns):
    plt.figure(figsize=(8, 7))
    sns.scatterplot(data=df, x="longitude", y="latitude", hue="cluster", palette="tab10", s=8, alpha=0.45)
    plt.title("Customer locations by segment")
    plt.tight_layout()
    plt.show()

display(
    df[["cluster", "cluster_name"]]
    .drop_duplicates()
    .sort_values("cluster")
    .reset_index(drop=True)
)

id_cluster = ucc.export_id_cluster(df, f"{DATA_DIR}/id_and_cluster.csv")
print("Exported rows:", len(id_cluster))
id_cluster.head()

