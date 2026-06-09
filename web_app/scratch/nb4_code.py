elif selected_page == "NB4 Characterisation":
    cluster_id_map = {0: "Vegetarians", 1: "Regulars", 2: "Wellness", 3: "Promoters", 4: "Loyalists", 5: "Families", 6: "Economizers", 7: "Techies"}

    st.markdown("""
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Cluster Characterisation</h2>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:32px;'>
      <div>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 4 — Cluster Characterisation</div>
        <p style='font-size:18px; color:#374151; line-height:1.9; margin:0 0 24px 0; text-align: justify;'>
          Notebook 4 operationalises the clustering model by profiling each segment. It identifies the distinct behavioural and spending traits that define the eight communities, assigns data-grounded business names, and validates these labels against demographic and geographic metadata. The goal is to translate abstract cluster assignments into clear, distinct customer personas.
        </p>
        <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:14px; margin-bottom:28px;'>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Segments named</div>
            <div style='font-size:22px; font-weight:800; color:#c94f38;'>8</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Chart views used</div>
            <div style='font-size:22px; font-weight:800; color:#c94f38;'>7</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Customers profiled</div>
            <div style='font-size:22px; font-weight:800; color:#c94f38;'>32,015</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Min views to name</div>
            <div style='font-size:22px; font-weight:800; color:#c94f38;'>3</div>
          </div>
        </div>
      </div>

      <div style='background:linear-gradient(145deg, #ffffff, #f9fafb); border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-top:20px;'>
        <div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 4 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb4-1" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>1) Imports & data loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-2" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>2) Segment sizes</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-3" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>3) Spend profile</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-4" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>4) Behavioural & demographic profile</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-5" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>5) Loyalty & metadata checks</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-6" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>6) Normalised comparison</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-7" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>7) Feature plots</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-8" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>8) Cluster interpretation</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-9" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>9) Geographic check</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb4-10" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>10) Final segment names & export</div></a>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("""
      <div id="nb4-1"></div><div id="nb4-3"></div><div id="nb4-4"></div><div id="nb4-5"></div><div id="nb4-6"></div><div id="nb4-7"></div><div id="nb4-8"></div><div id="nb4-9"></div><div id="nb4-10"></div>
      <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:8px;'>1) Imports & data loading - Naming protocol</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Business names are assigned only after the modelling stage is complete. A name is only confirmed when the same pattern appears consistently across at least three views: the spend deviation table, the radar plot, the spend profile heatmap, and the demographic/behavioural profile. "The final name of each segment is chosen only when the same pattern appears in more than one view." This prevents confirmation bias and ensures that names reflect stable, data-grounded patterns rather than single-chart impressions.</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Cluster sizes
    st.markdown("""
<div id="nb4-2" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">2) Segment sizes - Customer count per community</h2></div>
""", unsafe_allow_html=True)
    id_cluster_df = load_csv_data("id_and_cluster.csv")
    cluster_counts = id_cluster_df.groupby("cluster_name").size().reset_index(name="customers")
    cluster_counts = cluster_counts.sort_values("customers", ascending=False)
    cluster_size_chart = alt.Chart(cluster_counts).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
        x=alt.X("cluster_name:N", sort="-y", title="Segment"),
        y=alt.Y("customers:Q", title="Number of customers"),
        color=alt.Color("cluster_name:N", scale=alt.Scale(domain=list(SEGMENT_NAME_COLORS.keys()), range=list(SEGMENT_NAME_COLORS.values())), legend=None),
        tooltip=["cluster_name", alt.Tooltip("customers:Q", format=",")]
    ).properties(height=320)
    st.altair_chart(cluster_size_chart, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The distribution of customers across the eight communities is free from pathological size imbalance. No single community dominates the dataset and no community is too small to be actionable. This balance is a direct consequence of the outlier separation step in NB1: removing multivariate extremes before clustering produces a more homogeneous input space in which K-Means converges to more evenly populated centroids. The two largest segments (Regulars and Economizers) are also the most behaviourally moderate, which is consistent with a retail customer base where the majority of customers have unremarkable spending patterns. The three smallest segments (Techies, Promoters, Families) are the most behaviourally distinctive — their smaller size reflects how rare those specific patterns are in the population, not a modelling failure. Segment sizes inform campaign prioritisation: larger segments offer higher absolute reach, while smaller but more homogeneous segments offer higher targeting precision.</p>
</div>
""", unsafe_allow_html=True)

    # Interactive spend heatmap
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Normalised spend profile per cluster — interactive heatmap</div>
</div>
""", unsafe_allow_html=True)
    seg_spend_df = load_csv_data("segment_spend_profile.csv")
    spend_heat_cols = [c for c in seg_spend_df.columns if c.startswith("lifetime_spend_")]
    seg_spend_df["cluster"] = pd.to_numeric(seg_spend_df["cluster"], errors="coerce")
    seg_spend_df = seg_spend_df.dropna(subset=["cluster"])
    seg_spend_df = seg_spend_df.sort_values("cluster")
    seg_spend_df["segment_name"] = seg_spend_df["cluster"].astype(int).map(cluster_id_map)
    spend_matrix = seg_spend_df[spend_heat_cols].values.astype(float)
    col_min = spend_matrix.min(axis=0); col_max = spend_matrix.max(axis=0)
    col_range = col_max - col_min; col_range[col_range == 0] = 1
    spend_matrix_norm = (spend_matrix - col_min) / col_range
    spend_col_labels = [c.replace("lifetime_spend_", "").replace("_", " ").title() for c in spend_heat_cols]
    spend_heat_fig = px.imshow(spend_matrix_norm, x=spend_col_labels, y=seg_spend_df["segment_name"].tolist(),
        color_continuous_scale=list(SEGMENT_COLORS.values()), zmin=0, zmax=1, text_auto=".2f")
    spend_heat_fig.update_layout(margin=dict(l=80, r=20, t=60, b=80), height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(spend_heat_fig, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Each cell is normalised to [0, 1] across segments per column, so the darkest cell identifies the highest-spending segment in that category. Techies concentrate spending in electronics, technology, and videogames. Vegetarians over-index in vegetables and non-alcoholic drinks. Families show elevated spend across groceries and hygiene. Groceries show similar shading across nearly all segments, confirming that its exclusion from the clustering distance was correct — it adds little discriminative power. Alcohol and petfood show very low values across all segments, confirming their niche status in the customer base. Hover over any cell to see the normalised score; compare columns to identify which category most cleanly separates one segment from the rest.</p>
</div>
""", unsafe_allow_html=True)

    # Interactive behavioural heatmap
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Normalised behavioural profile per cluster — interactive heatmap</div>
</div>
""", unsafe_allow_html=True)
    info_unscaled_comm = load_csv_data("info_clustering_unscaled.csv")
    customer_segments_comm = load_csv_data("customer_segments.csv")
    merged_comm = info_unscaled_comm.merge(customer_segments_comm, on="customer_id", how="inner")
    behav_features = ["percentage_of_products_bought_promotion", "tenure", "total_children", "number_complaints"]
    behav_by_cluster = merged_comm.groupby("cluster")[behav_features].mean().reset_index()
    behav_by_cluster["segment_name"] = behav_by_cluster["cluster"].map(cluster_id_map)
    behav_by_cluster = behav_by_cluster.sort_values("cluster")
    behav_matrix = behav_by_cluster[behav_features].values.astype(float)
    b_min = behav_matrix.min(axis=0); b_max = behav_matrix.max(axis=0)
    b_range = b_max - b_min; b_range[b_range == 0] = 1
    behav_matrix_norm = (behav_matrix - b_min) / b_range
    behav_heat_fig = px.imshow(behav_matrix_norm, x=["Promo sensitivity", "Tenure (years)", "Total children", "Avg complaints"],
        y=behav_by_cluster["segment_name"].tolist(), color_continuous_scale=list(SEGMENT_COLORS.values()), zmin=0, zmax=1,
        text_auto=".2f")
    behav_heat_fig.update_layout(margin=dict(l=80, r=20, t=60, b=80), height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(behav_heat_fig, use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Promotional sensitivity varies strongly across segments: Promoters register the maximum value on this dimension, confirming that their defining trait is price-driven purchasing. Tenure separates long-term customers (Loyalists, Vegetarians) from newer cohorts (Regulars), supporting differentiated retention versus acquisition strategies. Total children most strongly characterises the Families segment — the highest value on this axis was one of the primary naming criteria. Complaints vary modestly across segments; where elevated, they reflect higher transaction frequency rather than systematic dissatisfaction. Together, these four dimensions provide a multi-axis profile that is more actionable for campaign design than spend data alone.</p>
</div>
""", unsafe_allow_html=True)

    # Community cards
    try:
        seg_summary = load_csv_data("segment_summary.csv")
        seg_meta_grid = {
            0: {"name": "Vegetarians", "desc": "Full-price, promotion-resistant shoppers. Lead with curation/quality framing rather than discounts.", "icon_idx": 0},
            1: {"name": "Regulars", "desc": "Active but newer, deal-aware shoppers. Strong targets for onboarding to the loyalty program.", "icon_idx": 1},
            2: {"name": "Wellness", "desc": "Quiet, low-maintenance, and low-complaint shoppers. Exert low friction and buy full price.", "icon_idx": 2},
            3: {"name": "Promoters", "desc": "The ultimate deal-seekers (+145% promo share). Perfect for price-led campaign stacking.", "icon_idx": 3},
            4: {"name": "Loyalists", "desc": "Highest LTV, highest tenure (13.6 years), and highest loyalty flag adoption (76.8%). Reward and protect.", "icon_idx": 4},
            5: {"name": "Families", "desc": "Large households (avg. 5.41 kids). Loyal without needing promotions. Target with bulk-buying bundles.", "icon_idx": 5},
            6: {"name": "Economizers", "desc": "Restrained, low-friction spenders who buy at baseline. NOT deal-chasers; value baseline pricing.", "icon_idx": 6},
            7: {"name": "Techies", "desc": "Small households buying high-value tech. Cleanest electronics and audio cross-sell campaign audience.", "icon_idx": 7}
        }
        cluster_images = {0:VEGETARIANS_URI,1:REGULARS_URI,2:WELLNESS_URI,3:PROMOTERS_URI,4:LOYALISTS_URI,5:FAMILIES_URI,6:ECONOMIZERS_URI,7:TECHIES_URI}
        st.markdown("<div style='font-size:20px; font-weight:700; color:#111827; margin-top:40px; margin-bottom:4px;'>Your 8 customer communities</div>", unsafe_allow_html=True)
        cards_list_html = []
        for idx, row in seg_summary.iterrows():
            c_id = int(row['cluster']); share = row['share_%']; custs = int(row['customers'])
            meta = seg_meta_grid.get(c_id, {"name": f"Cluster {c_id}", "desc": "No description.", "icon_idx": 0})
            img_uri = cluster_images.get(c_id, SLICES_URIS[c_id % len(SLICES_URIS)])
            cards_list_html.append(f"<div class='community-card'><div class='community-card-icon-container'><img src='{img_uri}' class='community-card-img' /></div><div><h3 class='community-card-title'>{meta['name']}</h3><div class='community-card-value'>{share:.1f}%</div><div class='community-card-sub'>{custs:,} customers</div><div class='community-card-desc'>{meta['desc']}</div></div><div class='community-card-arrow'>→</div></div>")
        st.markdown(f"<div class='communities-grid'>{''.join(cards_list_html)}</div>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading segment summary: {e}")

    st.markdown("""
<div style='margin-top:48px; margin-bottom:6px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;'>Notebook 4 — Segment profiling charts</div>
  <div style='font-size:20px; font-weight:800; color:#111827; margin-bottom:4px;'>Deep-dive characterisation — all charts from NB4</div>
</div>
""", unsafe_allow_html=True)

    # Spend profile heatmap (NB4 version)
    st.markdown("""
<div id="nb4-3" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">3) Spend profile heatmap</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Average lifetime spend per cluster (€)</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "spend_heatmap.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The spend profile heatmap shows raw average lifetime spend in euros per cluster-category pair, with colour normalised across clusters per column so that the darkest cell identifies the highest-spending segment in each category. Cell annotations show the actual euro values. Techies stand out strongly in electronics, technology, and videogames — confirming they are the technology-oriented segment and making them the priority audience for any cross-sell campaign targeting premium devices. Vegetarians dominate vegetables and non-alcoholic drinks, consistent with a health- and diet-conscious profile. Families show elevated spend across groceries and hygiene, reflecting large household purchasing patterns. Loyalists rank highly across multiple categories simultaneously, consistent with a long-tenure, broad-basket profile. Economizers show consistently low raw spend values across all categories, reflecting a restrained purchasing style — importantly, this is not driven by promotion sensitivity (their promo usage is near the median), but by genuinely lower absolute spending levels. This raw-values version of the heatmap complements the normalised plotly version shown above by revealing the actual scale differences between clusters, which the normalised view compresses.</p>
</div>
""", unsafe_allow_html=True)

    # Behavioural + demographic heatmap (NB4)
    st.markdown("""
<div id="nb4-4" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">4) Behavioural & demographic profile</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Z-scores by cluster</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "behavioural_heatmap.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>This heatmap captures non-spend dimensions: customer age, tenure as a customer, number of children at home, number of teenagers at home, number of complaints, stores visited, and promotional sensitivity. Z-scores allow direct comparison across variables with different units and scales. Families show the strongest positive deviation on the children and teens dimensions, which is the primary naming driver for this segment. Loyalists score highest on tenure, consistent with their long-standing relationship with the retailer. Promoters score by far the strongest positive z-score on promotional sensitivity (percentage of products bought on promotion), confirming that this is their defining and differentiating characteristic. Regulars and Economizers have relatively flat profiles across behavioural dimensions, which contributes to their lower distinctiveness in the z-score space — their differentiation comes from the spend profile rather than demographic or behavioural attributes. Complaints vary modestly across segments; no cluster is systematically dissatisfied, reducing the risk that any identified community represents a cohort at high churn risk due to service quality alone.</p>
</div>
""", unsafe_allow_html=True)

    # Individual radar profiles
    st.markdown("""
<div id="nb4-7" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">7) Feature plots</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>7.1) Individual radar profiles — all 8 clusters (9-axis spider chart)</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "radar_individual.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Each panel shows a single cluster's average profile across nine axes: electronics, vegetables, meat, fish, technology, petfood, videogames, hygiene, and promotional sensitivity. Values are normalised to [0, 1] relative to the dataset maximum for each axis, so the shape of each radar reflects relative spend intensity rather than absolute euros. Techies (C7) present a highly asymmetric shape with large extensions along electronics and technology and a very small promotional sensitivity arm, confirming they are full-price technology buyers. Vegetarians (C0) show a large extension along the vegetables axis with a near-zero promotional arm, consistent with quality-driven, full-price vegetable purchasers. Promoters (C3) show a large promotional sensitivity arm but a relatively flat spend profile across product categories, confirming that what defines this segment is how they buy rather than what they buy. Families (C5) show elevated hygiene and meat arms. Loyalists (C4) present a broadly extended shape across multiple arms, reflecting their high-basket, broad-category purchasing. The individual view makes segment-specific patterns clear without the visual complexity of the overlaid comparison.</p>
</div>
""", unsafe_allow_html=True)

    # Combined radar
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.2) Combined radar — all 8 clusters overlaid</div>
</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "radar_combined.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The combined radar overlays all eight cluster profiles in a single chart, enabling direct visual comparison of where each community stands relative to every other on the same axis. The chart reveals the concentration of most profiles near the centre for the majority of axes — confirming that most spending categories are at moderate or low levels for most customers — while a small number of clusters extend significantly outward on specific axes. This visual concentration pattern is the radar equivalent of the z-score heatmap's near-zero cells for moderate segments. The axes where clear separation occurs (electronics for Techies, vegetables for Vegetarians, promotional sensitivity for Promoters) are precisely the axes that carry the highest discriminative power in the clustering distance. The combined radar is particularly useful for campaign planning: any axis where the target segment extends furthest from the centre while others remain near the origin represents an opportunity for category-specific messaging with minimal audience overlap risk.</p>
</div>
""", unsafe_allow_html=True)

    # Feature barplots
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.3) Average spend per category by cluster — grouped bar charts</div>
</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "feature_barplots.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Each panel shows one product category, with bars coloured by cluster and the y-axis representing the average lifetime spend in euros. This view complements the heatmap by making absolute scale differences explicit: the electronics panel, for example, reveals that Techies spend roughly twice the next-highest cluster on electronics, while the vegetables panel shows a moderate but consistent advantage for Vegetarians. The fish panel shows that Wellness customers have a notably elevated fish spend relative to other moderate-spending segments, an insight that would be invisible in a normalised view but visible here because the absolute difference is meaningful. The petfood panel confirms that the petfood feature, while excluded from the clustering distance, does differentiate one cluster (Families) in the profiling stage. The videogames panel shows that Techies dominate this category, while most other segments spend near zero — making videogames the most concentrated single-segment category in the dataset and therefore the most targeted cross-sell opportunity available from any campaign built on these segments.</p>
</div>
""", unsafe_allow_html=True)

    # Boxplot grid
    st.markdown("""
<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.4) Key variable distributions by cluster — boxplot grid</div>
</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "boxplot_grid.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The boxplot grid shows the within-cluster distribution of four key variables: total lifetime spend, promotional purchase ratio, customer age, and tenure. Unlike mean-based heatmaps, boxplots expose the spread and skew within each cluster — information that is essential for assessing how targeted a campaign can realistically be. The total spend panel reveals that Loyalists have the highest median spend and also the widest interquartile range, meaning that this segment contains both very high and moderately high spenders. Promoters show a very narrow promotional sensitivity distribution clustered near 1.0, confirming that their defining characteristic is consistent, not occasional, promotion usage. The age panel reveals that Regulars and Techies skew younger while Loyalists and Families are older on average. Tenure follows a similar pattern: Loyalists have the longest tenure and the narrowest spread, while Regulars and Economizers are newer customers with wider tenure distributions, consistent with a more recently acquired and more heterogeneous cohort. These within-cluster distributions inform the confidence level with which each segment can be targeted: narrow distributions mean higher message precision; wide distributions mean a broader or tiered communication strategy is more appropriate.</p>
</div>
""", unsafe_allow_html=True)

    # Geographic scatter by cluster (NB4)
    st.markdown("""
<div id="nb4-9" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">9) Geographic check</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Geographic distribution by cluster — static scatter</div>
""", unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "geo_scatter.png"
    if _p.exists():
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2: st.image(str(_p), use_container_width=True)
    st.markdown("""
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The geographic scatter overlays all eight cluster labels on the latitude-longitude coordinate space, using the same colour palette as the radar and heatmap charts. Because geography was deliberately excluded from the clustering distance, any spatial pattern visible here is an emergent property of the behavioural segmentation rather than a modelling artefact. The chart reveals that clusters are not randomly mixed across space: Techies and Loyalists show a higher concentration in the central urban zone, consistent with the younger and higher-income urban customer profile identified in the geographic analysis notebook. Families are more evenly distributed across the suburban periphery, consistent with lower population density in residential areas outside the city centre. Promoters appear throughout the map with no strong geographic concentration, suggesting that price-sensitivity is a behavioural trait not constrained to a particular residential area. This geographic overlay is used as a final profiling validation step: if a cluster appeared concentrated exclusively in a single neighbourhood, that would raise a flag that geography had leaked into the segmentation through a correlated variable. The relatively mixed spatial distribution across clusters confirms that the model is capturing behavioural rather than geographic patterns.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("<h3 style='margin-top: 48px; margin-bottom: 24px;'>Explore Clustered Data</h3>", unsafe_allow_html=True)

    try:
        seg_summary = load_csv_data("segment_summary.csv")
        seg_spend = load_csv_data("segment_spend_profile.csv")
        seg_complaints = load_csv_data("segment_complaints_profile.csv")
        
        cluster_options = {
            0: "Cluster 0: Vegetarians",
            1: "Cluster 1: Regulars",
            2: "Cluster 2: Wellness",
            3: "Cluster 3: Promoters",
            4: "Cluster 4: Loyalists",
            5: "Cluster 5: Families",
            6: "Cluster 6: Economizers",
            7: "Cluster 7: Techies"
        }
        
        selected_cluster = st.selectbox("Select cluster to inspect", options=list(cluster_options.keys()), format_func=lambda x: cluster_options[x])
        cluster_color = SEGMENT_COLORS.get(selected_cluster, "#ea580c")
        
        row_summary = seg_summary[pd.to_numeric(seg_summary['cluster'], errors='coerce') == selected_cluster]
        row_spend = seg_spend[pd.to_numeric(seg_spend['cluster'], errors='coerce') == selected_cluster]
        row_complaints = seg_complaints[pd.to_numeric(seg_complaints['cluster'], errors='coerce') == selected_cluster]
