"""
utils_geographic.py

Helper functions for the geographic exploration of the customer base.
Keeping the map-building and spatial-analysis logic here lets the notebook
stay readable and call short, descriptive functions instead of repeating
long plotting blocks.
"""

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from folium.plugins import HeatMap, MarkerCluster


def coordinate_coverage(df, lat_col="latitude", lon_col="longitude"):
    """Report how many customers actually have usable coordinates."""
    valid = df[[lat_col, lon_col]].dropna()

    coverage = pd.Series({
        "total_customers": len(df),
        "with_coordinates": len(valid),
        "missing_coordinates": len(df) - len(valid),
        "coverage_percent": round(len(valid) / len(df) * 100, 1),
    })
    return coverage


def geographic_summary(df, lat_col="latitude", lon_col="longitude"):
    """Return a small describe-style table for the coordinate columns."""
    summary = df[[lat_col, lon_col]].agg(["min", "max", "mean", "std"]).round(4)
    return summary


def plot_scatter_distribution(df, lat_col="latitude", lon_col="longitude",
                              color="#1B4F72"):
    """Simple longitude/latitude scatter to see the overall footprint."""
    valid = df[[lat_col, lon_col]].dropna()

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(valid[lon_col], valid[lat_col], s=2, alpha=0.3, color=color)
    ax.set_title("Customer Geographic Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_density_hexbin(df, lat_col="latitude", lon_col="longitude",
                        gridsize=100, cmap="YlOrRd"):
    """Hexbin density map on a log scale to handle the dense city centre."""
    valid = df[[lat_col, lon_col]].dropna()

    fig, ax = plt.subplots(figsize=(14, 10))
    hb = ax.hexbin(valid[lon_col], valid[lat_col], gridsize=gridsize,
                   cmap=cmap, bins="log", alpha=0.8)
    ax.set_title("Customer Density Map (log scale)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    cbar = plt.colorbar(hb, ax=ax)
    cbar.set_label("Number of customers (log scale)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def find_densest_point(df, lat_col="latitude", lon_col="longitude", gridsize=50):
    """Return the (lat, lon) of the densest hexbin cell.

    Uses a throwaway hexbin (closed immediately) just to access the binned
    counts, then returns the centre of the busiest cell.
    """
    valid = df[[lat_col, lon_col]].dropna()

    hb = plt.hexbin(valid[lon_col], valid[lat_col], gridsize=gridsize)
    counts = hb.get_array()
    offsets = hb.get_offsets()
    plt.close()

    busiest = counts.argmax()
    lon, lat = offsets[busiest]
    return round(float(lat), 4), round(float(lon), 4)


def plotly_scatter_map(df, lat_col="latitude", lon_col="longitude", zoom=6):
    """Interactive open-street-map scatter (Plotly)."""
    valid = df[[lat_col, lon_col]].dropna()
    fig = px.scatter_mapbox(valid, lat=lat_col, lon=lon_col, zoom=zoom, height=450)
    fig.update_layout(mapbox_style="open-street-map",
                      margin=dict(l=0, r=0, t=0, b=0))
    return fig


def folium_cluster_map(df, lat_col="latitude", lon_col="longitude",
                       sample_size=5000, zoom_start=11, random_state=42):
    """Folium map with clustered markers.

    A sample is used by default because tens of thousands of markers make the
    map sluggish without changing the visual story.
    """
    valid = df[[lat_col, lon_col]].dropna()
    if sample_size and len(valid) > sample_size:
        valid = valid.sample(sample_size, random_state=random_state)

    center = [valid[lat_col].mean(), valid[lon_col].mean()]
    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")
    marker_cluster = MarkerCluster(name="Customer density").add_to(fmap)

    for lat, lon in zip(valid[lat_col], valid[lon_col]):
        folium.CircleMarker(location=[lat, lon], radius=3, color="#3186cc",
                            fill=True, fill_color="#3186cc",
                            fill_opacity=0.7).add_to(marker_cluster)
    return fmap


def folium_heatmap(df, lat_col="latitude", lon_col="longitude",
                   sample_size=10000, zoom_start=11, random_state=42):
    """Folium heatmap of customer locations."""
    valid = df[[lat_col, lon_col]].dropna()
    if sample_size and len(valid) > sample_size:
        valid = valid.sample(sample_size, random_state=random_state)

    center = [valid[lat_col].mean(), valid[lon_col].mean()]
    fmap = folium.Map(location=center, zoom_start=zoom_start, tiles="CartoDB positron")
    HeatMap(valid[[lat_col, lon_col]].values.tolist(), radius=10).add_to(fmap)
    return fmap


def filter_bounding_box(df, lat_range, lon_range,
                        lat_col="latitude", lon_col="longitude"):
    """Return the customers inside a lat/lon rectangle."""
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    mask = (df[lat_col].between(lat_min, lat_max) &
            df[lon_col].between(lon_min, lon_max))
    region = df[mask].copy()

    share = round(len(region) / len(df) * 100, 2)
    print(f"Customers in region: {len(region):,} ({share}% of total)")
    return region


def compare_region_to_global(region_df, full_df, feature_cols):
    """Compare a region's average against the global average per feature.

    Handy for describing a dense pocket (e.g. a university area): shows where
    that region over- or under-indexes relative to everyone else.
    """
    rows = []
    for col in feature_cols:
        if col not in full_df.columns:
            continue
        region_avg = region_df[col].mean()
        global_avg = full_df[col].mean()
        diff_pct = ((region_avg - global_avg) / global_avg * 100
                    if global_avg not in (0, np.nan) else np.nan)
        rows.append({
            "feature": col,
            "region_avg": round(region_avg, 2),
            "global_avg": round(global_avg, 2),
            "diff_vs_global_%": round(diff_pct, 1) if pd.notna(diff_pct) else np.nan,
        })

    result = pd.DataFrame(rows)
    return result.sort_values("diff_vs_global_%", ascending=False, key=abs)
