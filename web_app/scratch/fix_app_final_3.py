import sys

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix scroll
old_scroll = """        <script>
            var body = window.parent.document.querySelector(".main");
            if (body) {
                body.scrollTo(0, 0);
            }
            window.parent.scrollTo(0, 0);
        </script>"""

new_scroll = """        <script>
            setTimeout(function() {
                var mainElements = window.parent.document.querySelectorAll(".main, .block-container");
                mainElements.forEach(function(el) { el.scrollTo({top: 0, behavior: 'instant'}); });
                window.parent.scrollTo({top: 0, behavior: 'instant'});
            }, 100);
        </script>"""
content = content.replace(old_scroll, new_scroll)

# Reorder NB3
start_idx = content.find('elif selected_page == "NB3 Clustering":')
end_idx = content.find('elif selected_page == "NB4 Characterisation":')
nb3_content = content[start_idx:end_idx]

# We will construct the new nb3_content manually
new_nb3 = """elif selected_page == "NB3 Clustering":
    st.markdown('''
    <div style='margin-top: 0px; margin-bottom: 24px;'>
        <h2 style='font-size: 56px; font-weight: 800; color: #000000; margin: 0; letter-spacing: -0.03em;'>Clustering</h2>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
    <div style='margin-bottom:32px;'>
      <div>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>Notebook 3 — Clustering</div>
        <p style='font-size:18px; color:#374151; line-height:1.9; margin:0 0 24px 0; text-align: justify;'>
          Notebook 3 is the core modelling stage. It systematically evaluates combinations of feature sets, scalers, and cluster counts (k=6 to 10) to determine the partition that best captures the underlying behavioural diversity in the customer base. The process is fully transparent: every diagnostic — silhouette, elbow, dendrogram, and PCA/UMAP projection — is documented and benchmarked.
        </p>
        <div style='display:grid; grid-template-columns:repeat(auto-fit, minmax(130px, 1fr)); gap:14px; margin-bottom:28px;'>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Algorithm</div>
            <div style='font-size:22px; font-weight:800; color:#c94f38;'>K-Means</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>K selected</div>
            <div style='font-size:22px; font-weight:800; color:#c94f38;'>8</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Scaler</div>
            <div style='font-size:22px; font-weight:800; color:#c94f38;'>MinMax</div>
          </div>
          <div style='background:#f7e6e1; border:1px solid rgba(201, 79, 56, 0.25); border-radius:12px; padding:16px 18px; text-align:center;'>
            <div style='font-size:11px; font-weight:600; color:#7a6454; margin-bottom:4px; text-transform:uppercase; letter-spacing:0.08em;'>Feature set</div>
            <div style='font-size:16px; font-weight:800; color:#c94f38; line-height:1.2;'>spend + promo<br/>no groceries</div>
          </div>
        </div>
      </div>

      <div style='background:linear-gradient(145deg, #ffffff, #f9fafb); border:1px solid #e5e7eb; border-radius:12px; padding:20px; box-shadow:0 4px 6px -1px rgba(0,0,0,0.05); margin-top:20px;'>
        <div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 3 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb3-1" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>1) Imports & data loading</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-2" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>2) Outlier strategy check</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-3" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>3) Candidate feature sets & scalers</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-4" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>4) Diagnostics A - scalers & elbow</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-5" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>5) Diagnostics B - feature set & k grid</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-6" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>6) Model configuration</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-7" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>7) Model fitting & validation</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-8" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>8) Method benchmarks</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-9" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>9) Segment profiling</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-10" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>10) Reattach outliers & export</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb3-11" style="text-decoration:none;"><div style='font-size:13px; color:#374151; font-weight:500;'>11) Final modelling conclusion</div></a>
        </div>
      </div>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown('''
      <div id="nb3-1" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:8px;'>1) Imports & data loading</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Loading the core data libraries and the unscaled characterisation datasets prepared in Notebook 2.</div>
      </div>

      <div id="nb3-2" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:8px;'>2) Outlier strategy check</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Verifying the impact of the Isolation Forest outlier removal from Notebook 1 on the cluster structures. Ensuring extreme values do not disproportionately pull the centroids.</div>
      </div>

      <div id="nb3-3" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:8px;'>3) Candidate feature sets & scalers</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Different combinations of features (including/excluding groceries, including promotional sensitivity) and scaling techniques (MinMaxScaler vs RobustScaler) are formulated for systematic evaluation.</div>
      </div>
    ''', unsafe_allow_html=True)

    # 4) Diagnostics A
    st.markdown('''
<div id="nb3-4" style="margin-top:40px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">4) Diagnostics A - Scaler comparison & Elbow</h2></div>
''', unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "scaler_comparison.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Both MinMaxScaler and RobustScaler were tested. MinMaxScaler consistently produces higher silhouette scores at k=8 and shows a cleaner elbow in the curve.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "elbow_silhouette.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The silhouette peaks locally near k=4 and k=8, with k=8 providing the best balance between geometric separation and the number of actionable segments.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "ward_dendrogram.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)
    _p = IMAGENS_DIR / "charts" / "alt_dendrograms.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)


    # 5) Diagnostics B
    st.markdown('''
<div id="nb3-5" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">5) Diagnostics B - Silhouette score grid & Projections</h2></div>
''', unsafe_allow_html=True)
    _p = IMAGENS_DIR / "charts" / "silhouette_grid.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)
    _p = IMAGENS_DIR / "charts" / "silhouette_blades.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)
    _p = IMAGENS_DIR / "charts" / "pca_projection.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)
    _p = IMAGENS_DIR / "charts" / "umap_projection.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)
    _p = IMAGENS_DIR / "charts" / "tsne_projection.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)
    _p = IMAGENS_DIR / "charts" / "zscore_heatmap.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)

    # 6, 7, 8, 9
    st.markdown('''
      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:20px; margin-bottom:28px; margin-top:40px;'>
        <div>
          <div id="nb3-6" style='font-size:13px; font-weight:700; color:#111827; margin-bottom:12px;'>6) Model configuration & 7) Model fitting & validation</div>
          <div style='display:flex; flex-direction:column; gap:10px;'>
            <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;'>
              <div style='font-size:13px; font-weight:600; color:#111827;'>Why K=8?</div>
              <div style='font-size:15px; color:#6b7280; margin-top:3px;'>Values between 6 and 10 were compared. K=8 balances geometric separation with business interpretability.</div>
            </div>
            <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;'>
              <div style='font-size:13px; font-weight:600; color:#111827;'>Why exclude groceries?</div>
              <div style='font-size:15px; color:#6b7280; margin-top:3px;'>Most customers spend heavily on groceries regardless of segment. Groceries are kept for profiling.</div>
            </div>
          </div>
        </div>
        <div>
          <div id="nb3-8" style='font-size:13px; font-weight:700; color:#111827; margin-bottom:12px;'>8) Method benchmarks - Alternatives</div>
          <div style='display:flex; flex-direction:column; gap:10px;'>
            <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;'>
              <div style='font-size:13px; font-weight:600; color:#111827;'>DBSCAN — rejected</div>
              <div style='font-size:15px; color:#6b7280; margin-top:3px;'>Classified too many as noise.</div>
            </div>
            <div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:10px; padding:14px 16px;'>
              <div style='font-size:13px; font-weight:600; color:#111827;'>SOM & hierarchical Ward</div>
              <div style='font-size:15px; color:#6b7280; margin-top:3px;'>K-Means was confirmed as the strongest model.</div>
            </div>
          </div>
        </div>
      </div>

      <div id="nb3-9" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:8px;'>9) Segment profiling - Naming rationale</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Business names are assigned only after the modelling stage is complete.</div>
      </div>
      
      <div id="nb3-10" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
        <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:8px;'>10) Reattach outliers & export</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Outliers are merged back into the dataset and assigned to the closest existing cluster centroid to ensure all customers receive a segment label for downstream business use.</div>
      </div>

      <div id="nb3-11" style='padding:24px; border-radius:16px; background:#f9fafb; border:1px solid #e5e7eb; margin-top:32px;'>
        <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:10px;'>11) Final modelling conclusion</div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The final model is KMeans with k=8, MinMaxScaler, and the feature set "spend + promo no groceries". The characterisation of each segment is carried out in Notebook 4.</p>
      </div>
    ''', unsafe_allow_html=True)
    render_footer()
"""

content = content.replace(nb3_content, new_nb3)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
