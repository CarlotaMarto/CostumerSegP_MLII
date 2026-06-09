import matplotlib.pyplot as plt
import os
import sys

TARGET_DIR = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\imagens\charts"
os.makedirs(TARGET_DIR, exist_ok=True)

TITLE_TO_FILE = {
    "Scaler comparison": "scaler_comparison.png",
    "Elbow": "elbow_silhouette.png",
    "Ward dendrogram": "ward_dendrogram.png",
    "Alternative dendrograms": "alt_dendrograms.png",
    "Silhouette grid": "silhouette_grid.png",
    "Silhouette ù KMeans": "silhouette_blades.png",
    "PCA - KMeans": "pca_projection.png",
    "UMAP - KMeans": "umap_projection.png",
    "t-SNE - KMeans": "tsne_projection.png",
    "Absolute spend profile": "spend_heatmap.png",
    "Segment profile (standardised": "zscore_heatmap.png",
    "Behavioural and demographic": "behavioural_heatmap.png",
    "Segment radar profiles": "radar_individual.png",
    "Segment radar ù all clusters": "radar_combined.png",
    "Feature bars": "feature_barplots.png",
    "Boxplot grid": "boxplot_grid.png",
    "Customer locations": "geo_scatter.png",
}

def custom_show(*args, **kwargs):
    fig = plt.gcf()
    
    # Try to find a title
    title = ""
    for ax in fig.axes:
        if ax.get_title():
            title = ax.get_title()
            break
    if not title and fig._suptitle:
        title = fig._suptitle.get_text()
    
    filename = None
    for k, v in TITLE_TO_FILE.items():
        if k.lower() in title.lower() or (k == "Silhouette ù KMeans" and "silhouette" in title.lower() and "kmeans" in title.lower()):
            filename = v
            break
    
    if not filename:
        # Save anyway with a slug
        import re
        slug = re.sub(r'[^a-z0-9]', '_', title.lower())
        if not slug:
            slug = "untitled"
        filename = f"{slug}.png"
        
    filepath = os.path.join(TARGET_DIR, filename)
    print(f"Saving plot '{title}' -> {filename}")
    plt.savefig(filepath, bbox_inches="tight", pad_inches=0.4, dpi=150)
    plt.close('all')

plt.show = custom_show
import builtins
builtins.display = print

print("Running clustering script...")
sys.path.append(r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\clustering")
os.chdir(r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\clustering")
try:
    code = open("../clustering_script_utf8.py", encoding='utf-8-sig').read()
    exec(code, globals())
except Exception as e:
    print("Error in clustering script:", e)

print("Running characterization script...")
try:
    code = open("../char_script_utf8.py", encoding='utf-8-sig').read()
    exec(code, globals())
except Exception as e:
    print("Error in char script:", e)

print("Done generating charts.")
