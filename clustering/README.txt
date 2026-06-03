Clustering feature/scaler search notebook

Use this after running the updated preprocessing notebook that exports info_clustering_unscaled.csv.

Notebook:
- 02_clustering_feature_scaler_search.ipynb

Purpose:
- Test lifetime_spend_* feature spaces with MinMax/Robust/Standard scaling.
- Compare absolute-spend feature spaces with and without groceries.
- Choose k in the professor's requested range: 7 to 10.
- Produce silhouette, PCA, UMAP, hierarchical dendrograms, DBSCAN benchmark and model comparison.
