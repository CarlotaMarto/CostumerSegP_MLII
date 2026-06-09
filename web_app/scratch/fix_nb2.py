import sys
import os

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = 0
for i, l in enumerate(lines):
    if '<div id="nb2-1"></div><div id="nb2-3"></div><div id="nb2-insights"' in l:
        start_idx = i
        break

end_idx = 0
for i, l in enumerate(lines[start_idx:]):
    if 'st.subheader("4) Customer geographic distribution")' in l:
        end_idx = start_idx + i
        break

if start_idx == 0 or end_idx == start_idx:
    print("Could not find indices.")
    sys.exit(1)

new_code = """    <div id="nb2-1" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1) Imports & 2) Data loading</h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Data Preparation</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>The geographic coordinates (latitude and longitude) are loaded from the customer information dataset. Customers with missing geographic data are excluded from this specific spatial analysis to ensure mapping integrity.</div>
    </div>

    <div id="nb2-3" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">3) Basic geographic statistics</h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Coordinate range</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>The spatial bounding box confirms that all valid customer coordinates fall within the expected region. There are no erroneous locations (e.g. coordinates in the ocean) that require cleaning.</div>
    </div>

    <div id="nb2-insights" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">Insights from Geospatial Mapping</h2></div>

    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Behavioural differences: hotspot vs. rest of base</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>The hotspot shows a distinct behavioural profile even before clustering labels are applied. The strongest differences are in <strong>age, product diversity, number of complaints, store visits, total spend, and promotion usage</strong>. Hotspot customers are younger, more active, and more variety-seeking — consistent with a younger urban population, though the data does not confirm student status directly.</div>
    </div>

    <div style='border-left:3px solid #111827; padding-left:20px;'>
      <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Why geography is excluded from clustering</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Including geographic coordinates in the clustering distance would create spatially-defined groups — customers near each other in space would be forced into the same cluster regardless of their spending behaviour. The objective is to discover <em>behavioural</em> communities, not geographic ones. Geography is kept as a profiling tool: after clusters are fitted, the geographic distribution of each cluster is inspected as a validation and characterisation layer.</div>
    </div>
    \"\"\", unsafe_allow_html=True)

    customer_info = pd.read_csv(BASE_DIR / "datasets" / "customer_info.csv")
    customer_info = customer_info.dropna(subset=["latitude", "longitude"]).copy()
    customer_info["promo_ratio"] = customer_info["percentage_of_products_bought_promotion"] * 100
    customer_info["size_spend"] = customer_info["lifetime_total_distinct_products"].fillna(0) / 50

    st.markdown("<div id='nb2-4' style='margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;'><h2 style='font-size:24px; font-weight:800; color:#111827; margin:0;'>4) Customer geographic distribution</h2></div>", unsafe_allow_html=True)
"""

lines = lines[:start_idx] + [new_code] + lines[end_idx+1:]

with open(app_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
