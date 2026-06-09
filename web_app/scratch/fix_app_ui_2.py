import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# -----------------
# Fix NB4 titles
# -----------------
content = content.replace(
    "<div style='font-size:13px; font-weight:700; color:#111827;'>Spend profile heatmap — average lifetime spend per cluster (€)</div>",
    "<div id='nb4-3' style='font-size:13px; font-weight:700; color:#111827;'>3) Spend profile heatmap — average lifetime spend per cluster (€)</div>"
)

content = content.replace(
    "<div style='font-size:13px; font-weight:700; color:#111827;'>Behavioural and demographic profile heatmap — z-scores by cluster</div>",
    "<div id='nb4-4' style='font-size:13px; font-weight:700; color:#111827;'>4) Behavioural and demographic profile heatmap — z-scores by cluster</div>"
)

content = content.replace(
    "<div style='font-size:13px; font-weight:700; color:#111827;'>Individual radar profiles — all 8 clusters (9-axis spider chart)</div>",
    "<div id='nb4-7-1' style='font-size:13px; font-weight:700; color:#111827;'>7) Individual radar profiles — all 8 clusters (9-axis spider chart)</div>"
)

content = content.replace(
    "<div style='font-size:13px; font-weight:700; color:#111827;'>Combined radar — all 8 clusters overlaid</div>",
    "<div id='nb4-7-2' style='font-size:13px; font-weight:700; color:#111827;'>7) Combined radar — all 8 clusters overlaid</div>"
)

content = content.replace(
    "<div style='font-size:13px; font-weight:700; color:#111827;'>Average spend per category by cluster — grouped bar charts</div>",
    "<div id='nb4-7-3' style='font-size:13px; font-weight:700; color:#111827;'>7) Average spend per category by cluster — grouped bar charts</div>"
)

content = content.replace(
    "<div style='font-size:13px; font-weight:700; color:#111827;'>Key variable distributions by cluster — boxplot grid</div>",
    "<div id='nb4-7-4' style='font-size:13px; font-weight:700; color:#111827;'>7) Key variable distributions by cluster — boxplot grid</div>"
)

content = content.replace(
    "<div style='font-size:13px; font-weight:700; color:#111827;'>Geographic distribution by segment (customers per store)</div>",
    "<div id='nb4-9' style='font-size:13px; font-weight:700; color:#111827;'>9) Geographic check — Geographic distribution by segment (customers per store)</div>"
)

# -----------------
# Reorder NB5
# -----------------
# The current NB5 top grid contains 3, lift-derived, 7, 6.
# We want it to just contain 3 and lift-derived. Then we put 6 and 7 AFTER the charts.

old_nb5_grid = """      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:20px; margin-bottom:24px;'>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-3" style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>3) Apriori parameters - Why support is set at 1%</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Rules are mined per segment — each sub-population has far fewer transactions than the full dataset. A 1% support threshold ensures enough rules are discovered while still requiring meaningful co-occurrence frequency within each community.</div>
        </div>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Lift-derived campaign discounts</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Suggested campaign discounts are not fixed — they are derived from the lift value of each rule. A higher lift means a stronger-than-random co-purchase signal, which justifies a larger promotional incentive. This ties the commercial decision directly to statistical evidence.</div>
        </div>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-7" style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>7) Campaign suggestions - Excluded recommendations: Vegetarians (cluster 0)</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Chicken, meat, and fish are excluded from recommendations for cluster 0 (Vegetarians). The Apriori rules initially suggested these items, but they contradict the segment's defining behavioural trait. Notebook 4 confirms that this segment's identity is plant-based — the exclusion ensures campaign coherence.</div>
        </div>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-6" style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>6) Rule robustness check - Robustness validation</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Each segment's rules are validated on an 80/20 train/test split. Segments with many matched rules and low mean lift difference between train and test have stable co-purchase patterns. Segments with few matched rules should be interpreted with caution.</div>
        </div>
      </div>"""

new_nb5_grid_top = """      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:20px; margin-bottom:24px;'>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-3" style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>3) Apriori parameters - Why support is set at 1%</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Rules are mined per segment — each sub-population has far fewer transactions than the full dataset. A 1% support threshold ensures enough rules are discovered while still requiring meaningful co-occurrence frequency within each community.</div>
        </div>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>Lift-derived campaign discounts</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Suggested campaign discounts are not fixed — they are derived from the lift value of each rule. A higher lift means a stronger-than-random co-purchase signal, which justifies a larger promotional incentive. This ties the commercial decision directly to statistical evidence.</div>
        </div>
      </div>"""

content = content.replace(old_nb5_grid, new_nb5_grid_top)

# Now insert 6 and 7 after chart 2 (scatter_fig).
# Let's find: `st.markdown("<div style='height:1px; background:#e5e7eb; margin-bottom:24px;'></div>", unsafe_allow_html=True)`
# which is right after the interpretation of Chart 2.
insertion_point = """st.markdown("<div style='height:1px; background:#e5e7eb; margin-bottom:24px;'></div>", unsafe_allow_html=True)

    try:
        campaign_rules = load_csv_data("segment_campaign_rules.csv")"""

new_insertion = """st.markdown("<div style='height:1px; background:#e5e7eb; margin-bottom:24px;'></div>", unsafe_allow_html=True)
    st.markdown('''
      <div style='display:grid; grid-template-columns:repeat(2,1fr); gap:20px; margin-bottom:24px;'>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-6" style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>6) Rule robustness check - Robustness validation</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Each segment's rules are validated on an 80/20 train/test split. Segments with many matched rules and low mean lift difference between train and test have stable co-purchase patterns. Segments with few matched rules should be interpreted with caution.</div>
        </div>
        <div style='border-left:3px solid #111827; padding-left:20px;'>
          <div id="nb5-7" style='font-size:13px; font-weight:700; color:#111827; margin-bottom:6px;'>7) Campaign suggestions & Creative texts</div>
          <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Excluded recommendations: Chicken, meat, and fish are excluded from recommendations for cluster 0 (Vegetarians). The Apriori rules initially suggested these items, but they contradict the segment's defining behavioural trait. This ensures campaign coherence.</div>
        </div>
      </div>
    ''', unsafe_allow_html=True)
    
    try:
        campaign_rules = load_csv_data("segment_campaign_rules.csv")"""

content = content.replace(insertion_point, new_insertion)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated app.py UI for NB4 & NB5!")
