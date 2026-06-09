import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# -----------------
# Fix NB3 charts
# -----------------
rep_diag_a = """
    _p = IMAGENS_DIR / "charts" / "scaler_comparison.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>Scaler comparison</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Both MinMaxScaler and RobustScaler were tested. MinMaxScaler consistently produces higher silhouette scores at k=8 and shows a cleaner elbow in the curve.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "elbow_silhouette.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>Elbow method (Silhouette score)</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
    st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The silhouette peaks locally near k=4 and k=8, with k=8 providing the best balance between geometric separation and the number of actionable segments.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "ward_dendrogram.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>Ward Dendrogram</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The Ward dendrogram confirms that an 8-cluster cut perfectly aligns with the natural hierarchical structure of the data, showing well-separated, balanced branches.</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "alt_dendrograms.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>Alternative linkages (Average, Complete, Single)</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Alternative linkage methods exhibit severe chaining and pathological imbalance, confirming that Ward linkage (which minimises within-cluster variance) is the only viable approach for this dataset.</p>
</div>
''', unsafe_allow_html=True)
"""

content = re.sub(
    r'_p = IMAGENS_DIR / "charts" / "scaler_comparison\.png".*?if _p\.exists\(\): st\.image\(str\(_p\), use_container_width=True\)',
    rep_diag_a,
    content,
    flags=re.DOTALL
)

rep_projections = """
    _p = IMAGENS_DIR / "charts" / "silhouette_grid.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>Silhouette score grid (K=6 to K=10)</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
        
    _p = IMAGENS_DIR / "charts" / "silhouette_blades.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>Silhouette blades (K=8)</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>The blades for k=8 show relatively uniform thickness (indicating balanced cluster sizes) and mostly positive scores, with few instances of negative silhouette scores (which would indicate misclassification).</p>
</div>
''', unsafe_allow_html=True)

    _p = IMAGENS_DIR / "charts" / "pca_projection.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>PCA Projection (2D)</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)

    _p = IMAGENS_DIR / "charts" / "umap_projection.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>UMAP Projection (2D)</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)

    _p = IMAGENS_DIR / "charts" / "tsne_projection.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:13px; font-weight:700; color:#111827;'>t-SNE Projection (2D)</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>While PCA shows some overlap due to its linear nature, the manifold learning techniques (UMAP and particularly t-SNE) demonstrate clear boundaries and distinct islands, visually validating the K-Means cluster assignments in a non-linear low-dimensional space.</p>
</div>
''', unsafe_allow_html=True)
"""

content = re.sub(
    r'_p = IMAGENS_DIR / "charts" / "silhouette_grid\.png".*?_p = IMAGENS_DIR / "charts" / "tsne_projection\.png"\n    if _p\.exists\(\): st\.image\(str\(_p\), use_container_width=True\)',
    rep_projections,
    content,
    flags=re.DOTALL
)

# -----------------
# Fix NB5 Index wrapper
# -----------------
old_nb5_index = """<div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 5 Index</div>
          <a href="#nb5-1" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>1) Imports & data loading</div></a>"""

new_nb5_index = """<div style='font-size:14px; font-weight:800; color:#111827; margin-bottom:16px; border-bottom:2px solid #f3eee6; padding-bottom:8px;'>Notebook 5 Index</div>
        <div style='display:flex; flex-wrap:wrap; gap:12px; align-items:center;'>
          <a href="#nb5-1" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>1) Imports & data loading</div></a>"""

content = content.replace(old_nb5_index, new_nb5_index)

old_nb5_index_end = """<a href="#nb5-8" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>8) Final interpretation</div></a>
        </div>
      </div>
    </div>
    \"\"\", unsafe_allow_html=True)"""

new_nb5_index_end = """<a href="#nb5-8" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>8) Final interpretation</div></a>
        </div>
        </div>
      </div>
    </div>
    \"\"\", unsafe_allow_html=True)"""

content = content.replace(old_nb5_index_end, new_nb5_index_end)


old_nb5_leak = '''st.markdown(f"<div id='nb5-8'></div>\\n\\n#### Top Association Rules for {selected_segment}")'''
new_nb5_leak = '''st.markdown(f"<div id='nb5-8'></div>\\n\\n#### Top Association Rules for {selected_segment}", unsafe_allow_html=True)'''
content = content.replace(old_nb5_leak, new_nb5_leak)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated app.py successfully!")
