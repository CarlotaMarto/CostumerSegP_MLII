"""
Export script: Reads the existing CSVs from the data/ folder
and generates JSON files that the webapp can consume.

This script does NOT modify any existing data or code.
It only READS from data/ and WRITES to webapp/public/data/.
"""

import pandas as pd
import numpy as np
import json
import os
import sys

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'public', 'data')

os.makedirs(OUTPUT_DIR, exist_ok=True)


def export_dataset_overview():
    """Export general dataset statistics for the EDA section."""
    # Try final_dataset.csv first (modelling branch), then fall back to final_dataset_clean.csv
    final_path = os.path.join(DATA_DIR, 'final_dataset.csv')
    if not os.path.exists(final_path):
        final_path = os.path.join(DATA_DIR, 'final_dataset_clean.csv')
    df = pd.read_csv(final_path, index_col=0)
    
    overview = {
        "total_customers": int(len(df)),
        "total_features": int(len(df.columns)),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    
    # Numeric stats
    numeric_df = df.select_dtypes(include=[np.number])
    stats = {}
    for col in numeric_df.columns:
        stats[col] = {
            "mean": round(float(numeric_df[col].mean()), 2),
            "median": round(float(numeric_df[col].median()), 2),
            "std": round(float(numeric_df[col].std()), 2),
            "min": round(float(numeric_df[col].min()), 2),
            "max": round(float(numeric_df[col].max()), 2),
        }
    overview["numeric_stats"] = stats
    
    with open(os.path.join(OUTPUT_DIR, 'dataset_overview.json'), 'w') as f:
        json.dump(overview, f, indent=2)
    
    print(f"✓ dataset_overview.json ({len(df)} customers, {len(df.columns)} features)")
    return df


def export_demographics(df):
    """Export demographic distributions."""
    demographics = {}
    
    # Gender distribution
    if 'customer_gender' in df.columns:
        demographics['gender'] = df['customer_gender'].value_counts().to_dict()
    
    # Age distribution (binned)
    if 'customer_age' in df.columns:
        age_bins = [0, 25, 35, 45, 55, 65, 120]
        age_labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        age_groups = pd.cut(df['customer_age'], bins=age_bins, labels=age_labels)
        demographics['age_distribution'] = age_groups.value_counts().sort_index().to_dict()
    
    # Education level
    if 'education_level' in df.columns:
        demographics['education'] = df['education_level'].value_counts().to_dict()
    
    # Time of day
    if 'time_of_day' in df.columns:
        demographics['time_of_day'] = df['time_of_day'].value_counts().to_dict()
    
    # Children
    if 'total_children' in df.columns:
        demographics['total_children'] = df['total_children'].value_counts().sort_index().to_dict()
    elif 'has_children' in df.columns:
        demographics['has_children'] = {
            'With Children': int(df['has_children'].sum()),
            'Without Children': int(len(df) - df['has_children'].sum())
        }
        
    # Complaints
    if 'number_complaints' in df.columns:
        demographics['complaints'] = df['number_complaints'].value_counts().sort_index().to_dict()
    elif 'Number_of_Complaints' in df.columns:
        demographics['complaints'] = df['Number_of_Complaints'].value_counts().sort_index().to_dict()
    
    with open(os.path.join(OUTPUT_DIR, 'demographics.json'), 'w') as f:
        json.dump(demographics, f, indent=2)
    
    print(f"✓ demographics.json")


def export_spending(df):
    """Export spending patterns."""
    spend_cols = [col for col in df.columns if 'lifetime_spend' in col]
    
    spending = {}
    
    # Average spending per category
    avg_spending = {}
    for col in spend_cols:
        label = col.replace('lifetime_spend_', '').replace('_', ' ').title()
        avg_spending[label] = round(float(df[col].mean()), 2)
    spending['average_per_category'] = avg_spending
    
    # Total spending distribution (histogram bins)
    if spend_cols:
        total_spend = df[spend_cols].sum(axis=1).dropna()
        if len(total_spend) > 0:
            hist, bin_edges = np.histogram(total_spend, bins=20)
            spending['total_spend_histogram'] = {
                'counts': hist.tolist(),
                'bin_edges': [round(float(x), 2) for x in bin_edges.tolist()]
            }
            
            # Add log transformation for right-skewed data
            total_spend_log = np.log1p(total_spend)
            hist_log, bin_edges_log = np.histogram(total_spend_log, bins=20)
            spending['total_spend_log_histogram'] = {
                'counts': hist_log.tolist(),
                'bin_edges': [round(float(x), 2) for x in bin_edges_log.tolist()]
            }
        spending['total_spend_stats'] = {
            'mean': round(float(total_spend.mean()), 2),
            'median': round(float(total_spend.median()), 2),
            'std': round(float(total_spend.std()), 2),
            'min': round(float(total_spend.min()), 2),
            'max': round(float(total_spend.max()), 2),
        }
    
    with open(os.path.join(OUTPUT_DIR, 'spending.json'), 'w') as f:
        json.dump(spending, f, indent=2)
    
    print(f"✓ spending.json")


def export_geography(df):
    """Export geographic data (sampled for performance)."""
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # Sample to avoid massive JSON
        sample = df[['latitude', 'longitude', 'cluster_kmeans'] if 'cluster_kmeans' in df.columns else ['latitude', 'longitude']].dropna(subset=['latitude', 'longitude']).sample(
            min(12000, len(df)), random_state=42
        )
        
        points = []
        for _, row in sample.iterrows():
            pt = {'lat': round(float(row['latitude']), 4), 'lng': round(float(row['longitude']), 4)}
            if 'cluster_kmeans' in row and pd.notnull(row['cluster_kmeans']):
                pt['cluster'] = int(row['cluster_kmeans'])
            points.append(pt)
            
        geo = {'points': points}
        
        with open(os.path.join(OUTPUT_DIR, 'geography.json'), 'w') as f:
            json.dump(geo, f, indent=2)
        
        print(f"✓ geography.json ({len(geo['points'])} points)")


def export_basket_stats(df):
    """Export basket-related statistics."""
    basket_cols = ['total_trips', 'total_items_bought', 'average_basket_size', 
                   'max_basket_size', 'min_basket_size', 'unique_products_bought']
    
    available_cols = [col for col in basket_cols if col in df.columns]
    
    if available_cols:
        basket = {}
        for col in available_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                label = col.replace('_', ' ').title()
                basket[label] = {
                    'mean': round(float(col_data.mean()), 2),
                    'median': round(float(col_data.median()), 2),
                    'std': round(float(col_data.std()), 2),
                    'min': round(float(col_data.min()), 2),
                    'max': round(float(col_data.max()), 2),
                }
        
        # Histogram for total_trips
        if 'total_trips' in df.columns:
            valid_trips = df['total_trips'].dropna()
            if len(valid_trips) > 0:
                hist, edges = np.histogram(valid_trips, bins=15)
                basket['trips_histogram'] = {
                    'counts': hist.tolist(),
                    'bin_edges': [round(float(x), 2) for x in edges.tolist()]
                }
                
                # Add log transformation
                valid_trips_log = np.log1p(valid_trips)
                hist_log, edges_log = np.histogram(valid_trips_log, bins=15)
                basket['trips_log_histogram'] = {
                    'counts': hist_log.tolist(),
                    'bin_edges': [round(float(x), 2) for x in edges_log.tolist()]
                }
        
        with open(os.path.join(OUTPUT_DIR, 'basket_stats.json'), 'w') as f:
            json.dump(basket, f, indent=2)
        
        print(f"✓ basket_stats.json")


def export_preprocessing_summary():
    """Export before/after metrics from preprocessing."""
    raw = pd.read_csv(os.path.join(DATA_DIR, 'customer_info.csv'), index_col=0)
    
    # Try final_dataset.csv first (modelling branch), then fall back
    clean_path = os.path.join(DATA_DIR, 'final_dataset.csv')
    if not os.path.exists(clean_path):
        clean_path = os.path.join(DATA_DIR, 'final_dataset_clean.csv')
    clean = pd.read_csv(clean_path, index_col=0)
    
    summary = {
        "raw": {
            "rows": int(len(raw)),
            "columns": int(len(raw.columns)),
            "missing_values": int(raw.isnull().sum().sum()),
            "duplicates": int(raw.duplicated().sum()),
        },
        "clean": {
            "rows": int(len(clean)),
            "columns": int(len(clean.columns)),
            "missing_values": int(clean.isnull().sum().sum()),
            "duplicates": int(clean.duplicated().sum()),
        },
        "pipeline_steps": [
            {"name": "Data Type Handling", "description": "Converted dates, extracted customer age, cleaned numeric types"},
            {"name": "Duplicate Removal", "description": "Eliminated duplicate rows from both datasets"},
            {"name": "Impossible Values", "description": "Filtered age (0-120), negative counts, invalid coordinates"},
            {"name": "Missing Value Imputation", "description": "KNN Imputation (k=7) for numeric columns, zero-fill for spend columns"},
            {"name": "Outlier Detection", "description": "DBSCAN-based multidimensional outlier removal"},
            {"name": "Feature Engineering", "description": "Percentage-based spending features, education level extraction, family features, cyclic hour encoding"},
            {"name": "Basket Aggregation", "description": "Parsed shopping lists, computed basket metrics per customer"},
            {"name": "Dataset Merge", "description": "Inner join of customer info and basket features on customer_id"},
        ]
    }
    
    with open(os.path.join(OUTPUT_DIR, 'preprocessing.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ preprocessing.json")


def export_correlation_matrix(df):
    """Export correlation matrix for numeric features."""
    spend_cols = [col for col in df.columns if 'lifetime_spend' in col]
    behavior_cols = ['customer_age', 'total_children', 'distinct_stores_visited', 
                     'number_complaints', 'total_trips', 'average_basket_size',
                     'percentage_of_products_bought_promotion']
    
    selected = [col for col in spend_cols + behavior_cols if col in df.columns]
    
    if selected:
        corr = df[selected].corr()
        labels = [col.replace('lifetime_spend_', '').replace('_', ' ').title() for col in selected]
        
        correlation = {
            'labels': labels,
            'matrix': corr.round(3).values.tolist()
        }
        
        with open(os.path.join(OUTPUT_DIR, 'correlation.json'), 'w') as f:
            json.dump(correlation, f, indent=2)
        
        print(f"✓ correlation.json ({len(selected)} features)")


def export_clusters(df):
    """Export cluster-specific statistics for the interactive personas."""
    
    if 'cluster_kmeans' not in df.columns:
        print("Skipping export_clusters: 'cluster_kmeans' not found in dataset.")
        return

    cluster_names = {
        0: "Older Long-Tenure",
        1: "Large Families",
        2: "Clean and Green",
        3: 'Young Most Recent Customer',
        4: "Tech Enthusiasts",
        5: "Promotion Seekers",
        6: "Protein Lovers",
        7: "Service Sensitive"
    }

    features = [
        'customer_age', 'number_complaints', 'distinct_stores_visited',
        'total_children', 'percentage_of_products_bought_promotion',
        'lifetime_spend_vegetables', 'lifetime_spend_meat_fish',
        'lifetime_spend_electronics_videogames', 'lifetime_spend_nonalcohol_drinks',
        'lifetime_spend_alcohol_drinks', 'lifetime_spend_hygiene',
        'lifetime_spend_petfood', 'years_tenure'
    ]
    
    # Filter only features that exist in the dataframe
    valid_features = [f for f in features if f in df.columns]

    # Calculate global means for baseline comparison
    global_means = df[valid_features].mean().to_dict()

    clusters_data = {}
    total_customers = len(df)

    for cluster_id, group in df.groupby('cluster_kmeans'):
        if cluster_id not in cluster_names:
            continue
            
        cluster_size = len(group)
        cluster_dict = {
            "id": int(cluster_id),
            "name": cluster_names[int(cluster_id)],
            "size": cluster_size,
            "percentage": round((cluster_size / total_customers) * 100, 1),
            "features": {},
            "global_comparison": {}
        }

        # Calculate cluster means
        cluster_means = group[valid_features].mean().to_dict()
        
        for feature in valid_features:
            cluster_dict["features"][feature] = round(float(cluster_means[feature]), 2)
            # Calculate % difference from global mean
            if global_means[feature] > 0:
                diff_pct = ((cluster_means[feature] - global_means[feature]) / global_means[feature]) * 100
                cluster_dict["global_comparison"][feature] = round(float(diff_pct), 1)
            else:
                cluster_dict["global_comparison"][feature] = 0

        clusters_data[f"cluster_{cluster_id}"] = cluster_dict

    with open(os.path.join(OUTPUT_DIR, 'clusters.json'), 'w') as f:
        json.dump(clusters_data, f, indent=2)

    print(f"✓ clusters.json ({len(clusters_data)} personas)")


if __name__ == '__main__':
    print("=" * 50)
    print("Exporting data for webapp...")
    print("=" * 50)
    
    df = export_dataset_overview()
    
    # Merge clusters early so all exports can use them
    if 'cluster_kmeans' not in df.columns:
        clusters_path = os.path.join(DATA_DIR, 'customer_clusters.csv')
        if os.path.exists(clusters_path):
            clusters_df = pd.read_csv(clusters_path)
            if 'customer_id' in clusters_df.columns and 'customer_id' in df.columns:
                # Merge on customer_id
                df = df.merge(clusters_df[['customer_id', 'cluster_kmeans']], on='customer_id', how='inner')
    
    export_demographics(df)
    export_spending(df)
    export_geography(df)
    export_basket_stats(df)
    export_preprocessing_summary()
    export_correlation_matrix(df)
    export_clusters(df)
    
    print("=" * 50)
    print("All exports complete!")
    print(f"Files saved to: {os.path.abspath(OUTPUT_DIR)}")
