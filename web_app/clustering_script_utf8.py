import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

import utils_clustering as uc

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
pd.set_option("display.max_columns", None)

PROJECT_PALETTE = ['#B87540', '#B2543D', '#7E6A43', '#A8B7BA', '#D8C0B4', '#C8AB8C', '#5A3516', '#B98F70']
sns.set_theme(style='whitegrid', palette=PROJECT_PALETTE)
DATA_DIR = Path("../datasets")

regular = pd.read_csv(DATA_DIR / "info_clustering_unscaled.csv", index_col="customer_id")
outliers = pd.read_csv(DATA_DIR / "outlier_dataset.csv", index_col="customer_id")
raw_customer_info = pd.read_csv(DATA_DIR / "customer_info.csv")

print("Regular:", regular.shape, "| Outliers:", outliers.shape,
      "| Total:", len(regular) + len(outliers))
regular.head()

print('Technology split available:',
      [c for c in ['lifetime_spend_electronics', 'lifetime_spend_videogames'] if c in regular.columns])
capped = pd.read_csv(DATA_DIR / 'info_clustering_capped.csv', index_col='customer_id')
print('Capped dataset loaded:', capped.shape)
_COMPARE_FEATURE_SET = "spend + promo no groceries"
_COMPARE_K = 8
_COMPARE_SCALER = "MinMax"
_COMPARE_LOGABS = False
_candidate_sets = uc.build_candidate_feature_sets(regular)
_distance_cols, _logabs = _candidate_sets[_COMPARE_FEATURE_SET]
strategy_comparison = uc.compare_outlier_strategies(
    separation_df=regular,
    capped_df=capped,
    feature_cols=_distance_cols,
    k=_COMPARE_K,
    scaler_name=_COMPARE_SCALER,
    logabs=_COMPARE_LOGABS,
    random_state=0,
)
display(strategy_comparison)
uc.plot_strategy_umap_comparison(
    separation_df=regular,
    capped_df=capped,
    feature_cols=_distance_cols,
    k=_COMPARE_K,
    scaler_name=_COMPARE_SCALER,
    logabs=_COMPARE_LOGABS,
    random_state=0,
)
candidate_sets = uc.build_candidate_feature_sets(regular)
for name, (cols, logabs) in candidate_sets.items():
    print(f'{name:32s} | {len(cols):2d} cols | log_abs_spend={logabs}')

scaler_options = ['Standard', 'MinMax', 'Robust', 'None']
print('\nScalers available:', scaler_options)
INSPECT_SET = "spend + promo no groceries"  
INSPECT_SCALER = "MinMax"  

inspect_cols, inspect_logabs = candidate_sets[INSPECT_SET]
print(f'Inspecting: {INSPECT_SET}  ({len(inspect_cols)} cols, log_abs={inspect_logabs})')
print(f'Inspection scaler: {INSPECT_SCALER}')

scaler_scores = uc.compare_scalers(regular, inspect_cols, k_range=range(2, 14))
uc.plot_scaler_comparison(scaler_scores)
scaler_scores.pivot(index='k', columns='scaler', values='silhouette')
X_inspect = uc.apply_feature_pipeline(regular, inspect_cols, inspect_logabs,
                                      uc.get_scaler(INSPECT_SCALER), fit=True)
k_values, inertia = uc.kmeans_elbow(X_inspect, range(1, 14))
uc.plot_elbow(k_values, inertia)
DENDRO_CUT_HEIGHT = 6.4

uc.plot_sample_dendrogram(
    X_inspect,
    title=f"Ward dendrogram (sample) - {INSPECT_SET}",
    linkage="ward",
    sample_size=3000,
    cut_height=DENDRO_CUT_HEIGHT,
    random_state=0,
)
uc.plot_alternative_dendrograms(
    X_inspect,
    title_suffix=INSPECT_SET,
    k=8,
    sample_size=3000,
    random_state=0,
)

grid = uc.silhouette_grid(regular, candidate_sets, k_range=range(6, 11),
                          scaler_name='MinMax')
grid_pivot = uc.plot_silhouette_grid(grid, title='Silhouette grid (scaler = MinMax)')
grid_pivot
FEATURE_SET = "spend + promo no groceries"
SCALER      = "MinMax"  
K           = 8  

distance_cols, LOGABS = candidate_sets[FEATURE_SET]
profiling_cols = uc.get_profiling_features(regular, distance_cols)
scaler = uc.get_scaler(SCALER)
X = uc.apply_feature_pipeline(regular, distance_cols, LOGABS, scaler, fit=True)

print(f'Feature set : {FEATURE_SET}  ({len(distance_cols)} cols, log_abs={LOGABS})')
print(f'Scaler      : {SCALER}')
print(f'k           : {K}')
print(f'Matrix      : {X.shape}')
kmeans = uc.fit_kmeans(X, K)
regular['cluster'] = kmeans.labels_
uc.plot_cluster_sizes(regular, 'cluster')
uc.cluster_sizes(regular, 'cluster')
avg_sil = uc.plot_silhouette_blades(X, kmeans.labels_, title=f'Silhouette ù KMeans (k={K})')
print('Average silhouette:', round(avg_sil, 3))
X_emb, lab_emb = uc.subsample(X, kmeans.labels_, n=8000)
pca_emb = uc.embed_pca(X_emb)
uc.plot_embedding(pca_emb, lab_emb, title='PCA - KMeans Clusters', method_name='PCA')
umap_emb, method = uc.embed_umap(X_emb)
uc.plot_embedding(umap_emb, lab_emb, title='UMAP - KMeans Clusters', method_name=method)
tsne_emb = uc.embed_tsne(X_emb, perplexity=30, random_state=0)
uc.plot_embedding(tsne_emb, lab_emb, title='t-SNE - KMeans Clusters', method_name='t-SNE')
petfood_features = [
    "lifetime_spend_electronics",
    "lifetime_spend_videogames",
    "lifetime_spend_vegetables",
    "lifetime_spend_hygiene",
    "lifetime_spend_petfood",
    "percentage_of_products_bought_promotion",
    "lifetime_spend_meat",
    "lifetime_spend_fish",
]

pet_model, X_pet, pet_labels, pet_scaler, pet_metrics = uc.fit_kmeans_solution(
    regular,
    petfood_features,
    k=8,
    scaler_name=SCALER,
    iqr_k=None,
    random_state=42,
    n_init=30,
)

current_metrics = uc.clustering_metrics(X, kmeans.labels_, random_state=42)
current_metrics.update({
    "solution": "Final candidate",
    "feature_logic": FEATURE_SET,
    "scaler": SCALER,
    "iqr_k": None,
    "n_features": len(distance_cols),
})

pet_metrics.update({
    "solution": "Petfood check",
    "feature_logic": "granular spend with petfood",
})

comparison = pd.DataFrame([current_metrics, pet_metrics])[
    ["solution", "feature_logic", "n_features", "scaler", "iqr_k",
     "silhouette", "negative_silhouette_pct", "min_cluster_pct", "max_cluster_pct"]
]
comparison

pet_avg_sil = uc.plot_silhouette_blades(
    X_pet,
    pet_labels,
    title="Silhouette - petfood alternative (k=8)",
)
print("Average silhouette:", round(pet_avg_sil, 3))

X_pet_emb, lab_pet_emb = uc.subsample(X_pet, pet_labels, n=8000, random_state=42)
pet_umap, pet_method = uc.embed_umap(X_pet_emb, random_state=42)
uc.plot_embedding(
    pet_umap,
    lab_pet_emb,
    title="UMAP - petfood alternative",
    method_name=pet_method,
)

K_SEARCH_RANGE = range(2, 11)

method_benchmarks = uc.run_method_benchmarks(
    X,
    k_range=K_SEARCH_RANGE,
    random_state=0,
)

uc.display_method_benchmarks(method_benchmarks)
ward_sample_idx, ward_labels = uc.fit_hierarchical_sample(
    X,
    K,
    sample_size=5000,
    linkage="ward",
    random_state=0,
)
display(uc.compare_solutions(
    kmeans.labels_[ward_sample_idx],
    ward_labels,
    name_a="KMeans",
    name_b="Ward",
))
uc.plot_umap_label_comparison(
    X[ward_sample_idx],
    kmeans.labels_[ward_sample_idx],
    ward_labels,
    title_a="UMAP - KMeans labels on Ward sample",
    title_b="UMAP - Ward labels on same sample",
    random_state=0,
)
hierarchical_comparison, hierarchical_solution_tables = uc.compare_hierarchical_linkages(
    X,
    K,
    kmeans.labels_,
    sample_size=5000,
    random_state=0,
)

display(hierarchical_comparison)

for linkage_name, comparison_table in hierarchical_solution_tables.items():
    print(f"KMeans vs Agglomerative-{linkage_name}")
    display(comparison_table)
hierarchical_r2 = uc.hierarchical_r2_grid(
    X,
    k_range=range(6, 11),
    linkages=["ward", "complete", "average", "single"],
    sample_size=5000,
    random_state=0,
)

uc.plot_hierarchical_r2(hierarchical_r2)
display(hierarchical_r2.pivot(index="k", columns="linkage", values="r2"))

MICRO_K = 20
micro_kmeans, centroids_df, Z = uc.apply_centroid_ward_macro(
    regular,
    X,
    distance_cols,
    K,
    micro_k=MICRO_K,
    random_state=0,
)
macro_comparison = uc.compare_solutions(
    regular["cluster"].values,
    regular["macro_cluster"].values,
    name_a="Final KMeans",
    name_b="Centroid Ward macro",
)
display(macro_comparison)

uc.plot_silhouette_blades(
    X,
    regular["macro_cluster"].values,
    title=f"Silhouette - Hierarchical on KMeans centroids (k={K})",
)
uc.plot_macro_embeddings(
    X,
    regular["macro_cluster"].values,
    sample_size=8000,
    random_state=0,
)
dbscan_results, valid_dbscan, invalid_dbscan = uc.dbscan_benchmark_table(
    X,
    eps_values=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
                0.12, 0.14, 0.16, 0.18, 0.20, 0.25, 0.30, 0.35, 0.40],
    min_samples_values=[3, 5, 8, 10, 15, 20, 30, 40],
    sample_size=8000,
    random_state=0,
)

practical_dbscan = uc.practical_dbscan_candidates(
    dbscan_results,
    min_clusters=2,
    max_clusters=12,
    max_noise_pct=35,
)

print("Best valid DBSCAN settings by silhouette")
display(valid_dbscan.head(10))

print("Practical DBSCAN candidates after limiting the noise share")
display(practical_dbscan.head(10))

print("Parameter settings with one cluster or no valid silhouette")
display(invalid_dbscan.head(10))
RUN_DBSCAN = False
DBSCAN_EPS = 0.9
DBSCAN_MIN_SAMPLES = 10

dbscan_labels = uc.maybe_fit_dbscan(
    X,
    run=RUN_DBSCAN,
    eps=DBSCAN_EPS,
    min_samples=DBSCAN_MIN_SAMPLES,
)

if dbscan_labels is not None:
    regular["dbscan_cluster"] = dbscan_labels
SOM_GRID = (12, 12)
SOM_ITERATIONS = 1000

som_cols, X_som, som_curve, som_model = uc.run_som_diagnostic(
    regular,
    scaler_name=SCALER,
    grid=SOM_GRID,
    iterations=SOM_ITERATIONS,
    sample_size=12000,
    random_state=0,
)
regular["som_unit"] = uc.assign_and_plot_som(
    regular,
    som_model,
    X_som,
    som_cols,
    grid=SOM_GRID,
)
spend_cols = [c for c in regular.columns if c.startswith('lifetime_spend_')]
spend_profile = uc.profile_clusters(regular, 'cluster', spend_cols)
uc.plot_profile_heatmap(spend_profile, 'Absolute spend profile by cluster')
spend_profile

if "lifetime_spend_petfood" in spend_profile.columns:
    pet_check = spend_profile.drop(index="OVERALL", errors="ignore").copy()
    pet_check = pet_check[["lifetime_spend_petfood"]].round(2)
    display(pet_check.sort_values("lifetime_spend_petfood", ascending=False))

    print("Highest absolute petfood cluster:", pet_check["lifetime_spend_petfood"].idxmax())

key_cols = [
    "log_total_spend", "percentage_of_products_bought_promotion",
    "distinct_stores_visited", "lifetime_total_distinct_products",
    "tenure", "total_children", "number_complaints",
]

mixed_profile = uc.profile_clusters(regular, "cluster", key_cols)
uc.plot_profile_heatmap_z(mixed_profile,
                          title="Segment profile (standardised per feature)")

scaled_profile = uc.plot_segment_separation(
    regular,
    label_col="cluster",
)
complaints_profile = uc.complaint_summary(
    regular,
    label_col="cluster",
    complaint_col="number_complaints",
    threshold=2,
)
display(complaints_profile)

if {'latitude','longitude'}.issubset(regular.columns):
    plt.figure(figsize=(8, 7))
    sns.scatterplot(data=regular, x='longitude', y='latitude',
                    hue='cluster', palette=['#B87540', '#B2543D', '#7E6A43', '#A8B7BA', '#D8C0B4', '#C8AB8C', '#5A3516', '#B98F70'], s=8, alpha=0.5, legend='full')
    plt.title('Customer locations by segment (profiling only)')
    plt.show()
geo_profile = uc.geographic_profile(
    regular,
    label_col="cluster",
)
cluster_labels = pd.DataFrame({
    "cluster": sorted(regular["cluster"].unique())
})
display(cluster_labels)

segments = uc.reattach_outliers_and_export(
    regular,
    outliers,
    kmeans,
    distance_cols,
    LOGABS,
    scaler,
    DATA_DIR,
    spend_profile=spend_profile,
    complaints_profile=complaints_profile if "complaints_profile" in locals() else None,
)
segments.head()
decision_log = pd.DataFrame({
    "parameter": ["Feature set", "Scaler", "k", "Log-abs transform"],
    "value":     [FEATURE_SET,   SCALER,   K,   LOGABS],
}).set_index("parameter")

metrics_log = pd.DataFrame({
    "metric": ["Silhouette"],
    "value":  [round(avg_sil, 4)],
}).set_index("metric")

display(decision_log)
display(metrics_log)
