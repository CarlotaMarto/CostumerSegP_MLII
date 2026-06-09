import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Standardise subtitle font size
content = content.replace("font-size:13px", "font-size:18px")

# 2. Add zscore_heatmap.png description and title
old_zscore = """    _p = IMAGENS_DIR / "charts" / "zscore_heatmap.png"
    if _p.exists(): st.image(str(_p), use_container_width=True)"""

new_zscore = """    _p = IMAGENS_DIR / "charts" / "zscore_heatmap.png"
    if _p.exists():
        st.markdown("<div style='margin-bottom:12px;'><div style='font-size:18px; font-weight:700; color:#111827;'>Z-Score Heatmap (Features x Clusters)</div></div>", unsafe_allow_html=True)
        st.image(str(_p), use_container_width=True)
        st.markdown('''
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <div style='font-size:11px; font-weight:700; letter-spacing:0.10em; text-transform:uppercase; color:#9ca3af; margin-bottom:8px;'>Interpretation</div>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>This heatmap visualises the z-score deviations of each cluster's centroid from the global mean across all features. Deep rust indicates values significantly above the average, while slate indicates values below the average. It confirms the distinct spending and behavioural profile of each segment: Techies dominate electronics and technology, Vegetarians dominate vegetables, and Families show strong deviations in meat and hygiene.</p>
</div>
''', unsafe_allow_html=True)"""

content = content.replace(old_zscore, new_zscore)

# 3. Constrain image sizes (wrap all `st.image(str(_p), use_container_width=True)` in a column)
lines = content.split('\n')
new_lines = []
for line in lines:
    if 'st.image(str(_p), use_container_width=True)' in line:
        indent = line[:len(line) - len(line.lstrip())]
        new_lines.append(indent + "col1, col2, col3 = st.columns([1, 8, 1])")
        # If it was an inline if, split it
        if line.strip().startswith('if '):
            parts = line.split(': ', 1)
            new_lines[-1] = indent + parts[0] + ":"
            new_lines.append(indent + "    col1, col2, col3 = st.columns([1, 8, 1])")
            new_lines.append(indent + "    with col2: " + parts[1])
        else:
            new_lines.append(indent + "with col2: st.image(str(_p), use_container_width=True)")
    else:
        new_lines.append(line)

content = '\n'.join(new_lines)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated app.py UI for final fixes!")
