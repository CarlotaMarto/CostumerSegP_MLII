import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update the first correlation chart
old_corr1 = """    corr_fig = px.imshow(
        corr_matrix.values,
        x=corr_labels,
        y=corr_labels,
        color_continuous_scale=list(SEGMENT_COLORS.values()),
        zmin=-1,
        zmax=1,
        text_auto=".2f"
    )"""

new_corr1 = """    custom_corr_scale = ["#6B7D7D", "#fcfbf8", "#9D5C4A"]
    corr_fig = px.imshow(
        corr_matrix.values,
        x=corr_labels,
        y=corr_labels,
        color_continuous_scale=custom_corr_scale,
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1,
        text_auto=".2f"
    )"""
content = content.replace(old_corr1, new_corr1)

# 2. Update the second correlation chart
old_corr2_block = """    corr_chart_path = IMAGENS_DIR / "charts" / "correlation_heatmap.png"
    if corr_chart_path.exists():
        st.image(str(corr_chart_path), use_container_width=True)"""

new_corr2_block = """    import numpy as np
    numeric_df = info_unscaled.select_dtypes(include=['number'])
    if 'customer_id' in numeric_df.columns:
        numeric_df = numeric_df.drop(columns=['customer_id'])
    full_corr = numeric_df.corr()
    mask = np.triu(np.ones_like(full_corr, dtype=bool))
    full_corr_masked = full_corr.mask(mask)
    
    full_corr_labels = [c.replace('_', ' ').title() for c in full_corr.columns]
    
    custom_corr_scale = ["#6B7D7D", "#fcfbf8", "#9D5C4A"]
    full_corr_fig = px.imshow(
        full_corr_masked.values,
        x=full_corr_labels,
        y=full_corr_labels,
        color_continuous_scale=custom_corr_scale,
        color_continuous_midpoint=0,
        zmin=-1,
        zmax=1
    )
    full_corr_fig.update_layout(margin=dict(l=60, r=20, t=60, b=100), height=800, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(full_corr_fig, use_container_width=True)"""

content = content.replace(old_corr2_block, new_corr2_block)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Applied custom palette and dynamic full heatmap.")
