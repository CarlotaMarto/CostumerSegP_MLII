import sys
import os

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

start_idx = 0
for i, l in enumerate(lines):
    if '<div id="nb0-1"></div><div id="nb0-1-1">' in l:
        start_idx = i
        break

end_idx = 0
for i, l in enumerate(lines):
    if '# Chart 5: Skewness table' in l:
        end_idx = i
        break

new_code = """    <div id="nb0-1"></div>
    <div id="nb0-1-1" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.1 Initial Data Analysis</h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Dataset Overview</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>The dataset contains demographic and transactional records for a subset of the customer base. This initial pass focuses on identifying data quality issues, structural problems, and general feature distributions before any preprocessing or scaling is applied.</div>
    </div>

    <div id="nb0-1-2" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.2 Duplicate Rows Analysis & <span id="nb0-1-2-1">1.2.1 Surname Repetition Check</span></h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Duplicate check</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>No exact duplicate rows were found. A logical duplicate check (matching on customer name AND birthdate) was also performed. A surname-only proximity test was also run but produced too many false positives due to common surnames. The conclusion is that the dataset does not contain systematic duplicate records requiring removal.</div>
    </div>

    <div id="nb0-1-3" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.3 Missing Values Analysis</h2></div>
    <div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
      <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Missing value strategy</div>
      <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Features with more than 30% missing values were flagged as too sparse to impute reliably. The inspection confirmed that missing values are concentrated in a limited group of behavioural and spend variables, supporting imputation over row-dropping — the customer base does not need to be reduced.</div>
    </div>
    \"\"\", unsafe_allow_html=True)

    customer_info = pd.read_csv(BASE_DIR / "datasets" / "customer_info.csv")

    # Chart 1: Missing values per feature
    st.markdown(\"\"\"
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:13px; font-weight:700; color:#111827;'>Missing values per feature (%)</div>
</div>
\"\"\", unsafe_allow_html=True)
    missing_pct = (customer_info.isnull().mean() * 100).reset_index()
    missing_pct.columns = ["feature", "missing_pct"]
    missing_pct = missing_pct.sort_values("missing_pct", ascending=False)
    missing_pct = missing_pct[missing_pct["missing_pct"] > 0]
    base_missing = alt.Chart(missing_pct).mark_bar(color=SEGMENT_COLORS[7], cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        x=alt.X("missing_pct:Q", title="Missing (%)"),
        tooltip=["feature", alt.Tooltip("missing_pct:Q", format=".2f", title="Missing %")]
    ).properties(height=max(200, len(missing_pct) * 22))
    threshold_line = alt.Chart(pd.DataFrame({"threshold": [30]})).mark_rule(color="#ef4444", strokeDash=[6, 3], strokeWidth=2).encode(
        x="threshold:Q"
    )
    st.altair_chart((base_missing + threshold_line), use_container_width=True)
    st.markdown(\"\"\"
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The 30% threshold (red dashed line) was chosen as the boundary above which imputation is judged unreliable: reconstructing more than three out of ten values for a given feature would introduce more noise than signal into the dataset. Features below this threshold retain sufficient observed data to support KNN imputation, which leverages the similarity structure of the customer base. The chart confirms that missing values are concentrated in a small number of behavioural variables, and that no feature exceeds the threshold by a large margin, making row-dropping unnecessary. The majority of the 33,038 customers remain usable across all features.</p>
</div>
\"\"\", unsafe_allow_html=True)

    # 1.4 Numerical and Categorical Columns
    st.markdown(\"\"\"
<div id="nb0-1-4" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.4 Numerical and Categorical Columns</h2></div>
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:13px; font-weight:700; color:#111827;'>Distribution of key numerical features</div>
</div>
\"\"\", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.markdown("<div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:8px;'>Lifetime spend: Groceries</div>", unsafe_allow_html=True)
    c2.markdown("<div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:8px;'>Lifetime spend: Electronics</div>", unsafe_allow_html=True)
    c3.markdown("<div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:8px;'>Total distinct products</div>", unsafe_allow_html=True)

    hist_groceries = alt.Chart(customer_info.dropna(subset=["lifetime_spend_groceries"])).mark_bar(color=SEGMENT_COLORS[3], opacity=0.85).encode(
        x=alt.X("lifetime_spend_groceries:Q", bin=alt.Bin(maxbins=30), title="Groceries spend"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=280)
    hist_electronics = alt.Chart(customer_info.dropna(subset=["lifetime_spend_electronics"])).mark_bar(color=SEGMENT_COLORS[5], opacity=0.85).encode(
        x=alt.X("lifetime_spend_electronics:Q", bin=alt.Bin(maxbins=30), title="Electronics spend"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=280)
    hist_products = alt.Chart(customer_info.dropna(subset=["lifetime_total_distinct_products"])).mark_bar(color=SEGMENT_COLORS[4], opacity=0.85).encode(
        x=alt.X("lifetime_total_distinct_products:Q", bin=alt.Bin(maxbins=30), title="Distinct products"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=280)
    c1.altair_chart(hist_groceries, use_container_width=True)
    c2.altair_chart(hist_electronics, use_container_width=True)
    c3.altair_chart(hist_products, use_container_width=True)
    st.markdown(\"\"\"
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>All three distributions exhibit pronounced right-skew: the mass of customers clusters near the lower end of the scale, with a progressively thinner tail extending toward high-spending or high-variety individuals. This asymmetry has two direct implications for modelling. First, standard Euclidean distance in clustering is sensitive to scale differences, meaning that a small group of high-spending customers could disproportionately pull cluster centroids if the data are not scaled. Second, the long tail is precisely where the consensus outlier separation strategy intervenes: rather than capping values, the most extreme multivariate observations are separated into a dedicated outlier dataset before the clustering model is fitted, preserving the shape of the majority distribution while removing undue influence from the periphery.</p>
</div>
\"\"\", unsafe_allow_html=True)

    # 1.4.1 Findings in Categorical Columns
    st.markdown(\"\"\"
<div id="nb0-1-4-1" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.4.1 Findings in Categorical Columns</h2></div>
<div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:20px;'>
  <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Education level as a proxy feature</div>
  <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Customer names contain academic prefixes — BSc., MSc., PhD. — across all 33,038 unique names. These prefixes are flagged as an education-level proxy and earmarked for feature engineering in Notebook 1. Surname repetition alone was found to be too common to be a useful household signal; it was not carried into the modelling feature set.</div>
</div>
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:13px; font-weight:700; color:#111827;'>Gender distribution</div>
</div>
\"\"\", unsafe_allow_html=True)
    gender_counts = customer_info["customer_gender"].value_counts().reset_index()
    gender_counts.columns = ["customer_gender", "count"]
    gender_bar = alt.Chart(gender_counts).mark_bar(color=SEGMENT_COLORS[6], cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("customer_gender:N", title="Gender"),
        y=alt.Y("count:Q", title="Number of customers"),
        tooltip=["customer_gender", alt.Tooltip("count:Q", title="Customers", format=",")]
    ).properties(height=300)
    st.altair_chart(gender_bar, use_container_width=True)
    st.markdown(\"\"\"
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The gender distribution across the customer base is approximately balanced between male and female customers, with no category representing an extreme minority. This near-parity is relevant for segmentation methodology: a heavily skewed gender distribution would risk producing segments that reflect gender composition artefacts rather than genuine behavioural differences. The approximate balance observed here supports the interpretation that the eight clusters recovered by the model reflect spending behaviour and lifestyle patterns rather than demographic overrepresentation of one group. Gender is retained as a profiling variable for segment characterisation but is not included in the clustering distance matrix.</p>
</div>
\"\"\", unsafe_allow_html=True)

    # 1.5 Statistical Summary
    st.markdown(\"\"\"
<div id="nb0-1-5" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1.5 Statistical Summary</h2></div>
<div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:8px;'>
  <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Impossible values detected</div>
  <div style='font-size:16px; color:#6b7280; line-height:1.8;'><code>percentage_of_products_bought_promotion</code> was found to contain values outside the valid [0, 1] range — both below 0.0 and above 1.0 — indicating data entry errors. These are flagged here and corrected in preprocessing. Spending variables show strong right-skew, confirming that a small group of customers spends disproportionately more than the majority.</div>
</div>
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:13px; font-weight:700; color:#111827;'>Promotion ratio distribution (valid range only)</div>
</div>
\"\"\", unsafe_allow_html=True)
    promo_valid = customer_info[
        (customer_info["percentage_of_products_bought_promotion"] >= 0) &
        (customer_info["percentage_of_products_bought_promotion"] <= 1)
    ].copy()
    promo_hist = alt.Chart(promo_valid).mark_bar(color=SEGMENT_COLORS[1], opacity=0.85).encode(
        x=alt.X("percentage_of_products_bought_promotion:Q", bin=alt.Bin(maxbins=30), title="Proportion of products bought on promotion"),
        y=alt.Y("count():Q", title="Customers"),
        tooltip=[alt.Tooltip("count():Q", title="Customers")]
    ).properties(height=300)
    st.altair_chart(promo_hist, use_container_width=True)
    st.markdown(\"\"\"
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The raw dataset contains entries for <code>percentage_of_products_bought_promotion</code> that fall outside the physically valid interval [0, 1], including both negative values and values exceeding 1.0. These are impossible by definition: a proportion cannot be negative or greater than unity, confirming data entry errors rather than extreme but valid behaviour. The chart above is restricted to the valid range only. Within [0, 1], the distribution is roughly bimodal: a concentration of customers near 0.4 to 0.6 suggests a moderately promotion-responsive majority, while a second mass near 1.0 identifies a distinct group of near-exclusively promotional buyers. This heterogeneity in promotional sensitivity later becomes one of the most discriminating variables in the clustering model, most clearly visible in the Promoters segment.</p>
</div>
\"\"\", unsafe_allow_html=True)
\n"""

lines = lines[:start_idx] + [new_code] + lines[end_idx:]

with open(app_path, "w", encoding="utf-8") as f:
    f.writelines(lines)
