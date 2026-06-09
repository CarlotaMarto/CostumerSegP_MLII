import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Test inputs (approximately average values)
user_spends = {
    "lifetime_spend_groceries": 15843,
    "lifetime_spend_electronics": 2646,
    "lifetime_spend_vegetables": 730,
    "lifetime_spend_nonalcohol_drinks": 456,
    "lifetime_spend_alcohol_drinks": 605,
    "lifetime_spend_meat": 709,
    "lifetime_spend_fish": 593,
    "lifetime_spend_hygiene": 817,
    "lifetime_spend_videogames": 358,
    "lifetime_spend_petfood": 333,
    "lifetime_spend_technology": 3004
}
val_complaints = 1

df_spend_sim = pd.read_csv(BASE_DIR / "datasets" / "segment_spend_profile.csv")
df_comp_sim = pd.read_csv(BASE_DIR / "datasets" / "segment_complaints_profile.csv")

overall_spend = df_spend_sim[df_spend_sim['cluster'] == 'OVERALL'].iloc[0]
overall_complaints = (df_comp_sim['avg_complaints'] * df_comp_sim['customers']).sum() / df_comp_sim['customers'].sum()

print(f"Overall Spend Groceries: {overall_spend['lifetime_spend_groceries']}")
print(f"Overall Complaints: {overall_complaints:.4f}")

min_dist = float('inf')
best_cluster = 0
distances = {}

for c in range(8):
    c_spend = df_spend_sim[pd.to_numeric(df_spend_sim['cluster'], errors='coerce') == c].iloc[0]
    c_comp = df_comp_sim[pd.to_numeric(df_comp_sim['cluster'], errors='coerce') == c].iloc[0]
    
    dist_sq = 0.0
    
    for col_name in user_spends.keys():
        mu_overall = float(overall_spend[col_name])
        mu_cluster = float(c_spend[col_name])
        u_val = float(user_spends[col_name])
        if mu_overall > 0:
            dist_sq += ((u_val - mu_cluster) / mu_overall) ** 2
            
    u_comp = float(val_complaints)
    mu_c_comp = float(c_comp['avg_complaints'])
    if overall_complaints > 0:
        dist_sq += ((u_comp - mu_c_comp) / overall_complaints) ** 2
        
    dist = dist_sq ** 0.5
    distances[c] = dist
    print(f"Cluster {c} distance: {dist:.4f}")
    if dist < min_dist:
        min_dist = dist
        best_cluster = c

print(f"Best Cluster: {best_cluster} with distance {min_dist:.4f}")
