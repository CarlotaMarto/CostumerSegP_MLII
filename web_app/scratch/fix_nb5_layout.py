import re
import os

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# === FIX NB5 ===
start_nb5 = content.find('elif selected_page == "Targeter Promotion":')
end_nb5 = content.find('elif selected_page == "Conclusion & Recommendations":')
nb5_raw = content[start_nb5:end_nb5]

# Update the Index HTML in NB5 to include point 9
index_old = """          <a href="#nb5-8" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>8) Final interpretation</div></a>
        </div>"""

index_new = """          <a href="#nb5-8" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>8) Final interpretation</div></a>
          <div style='color:#d1d5db;'>•</div>
          <a href="#nb5-9" style="text-decoration:none;"><div style='font-size:12px; color:#374151; font-weight:500;'>9) Cupons</div></a>
        </div>"""
nb5_raw = nb5_raw.replace(index_old, index_new)

# Now rebuild the layout of NB5 sequentially.
# The original code has:
# 1) Intro & Index
# 2) Text boxes (3, Lift, 7, 6) -> out of order
# 3) Chart 4
# 4) Chart 5
# 5) Targeter 8
intro_part = nb5_raw[:nb5_raw.find("    st.markdown(\"\"\"\n      <div id=\"nb5-1\"></div><div id=\"nb5-2\"></div>")]

# We construct the sections
nb5_1_2 = """
    st.markdown('''
<div id="nb5-1" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">1) Imports & data loading</h2></div>
<div style='font-size:16px; color:#6b7280; line-height:1.8; margin-bottom:24px;'>Initialisation of the environment, loading the customer transactions and cluster assignments.</div>

<div id="nb5-2" style="margin-top:32px; margin-bottom:24px; border-top:1px solid #e5e7eb; padding-top:24px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">2) Transaction preparation per segment</h2></div>
<div style='font-size:16px; color:#6b7280; line-height:1.8; margin-bottom:24px;'>Separating the global transactions into segment-specific baskets to run Apriori per community.</div>
''', unsafe_allow_html=True)
"""

nb5_3 = """
    st.markdown('''
<div id="nb5-3" style="margin-top:32px; margin-bottom:24px; border-top:1px solid #e5e7eb; padding-top:24px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">3) Apriori parameters</h2></div>
<div style='display:grid; grid-template-columns:repeat(2,1fr); gap:20px; margin-bottom:24px;'>
    <div style='border-left:3px solid #111827; padding-left:20px;'>
        <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Why support is set at 1%</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Rules are mined per segment — each sub-population has far fewer transactions than the full dataset. A 1% support threshold ensures enough rules are discovered while still requiring meaningful co-occurrence frequency within each community.</div>
    </div>
    <div style='border-left:3px solid #111827; padding-left:20px;'>
        <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Lift-derived campaign discounts</div>
        <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Suggested campaign discounts are not fixed — they are derived from the lift value of each rule. A higher lift means a stronger-than-random co-purchase signal, which justifies a larger promotional incentive. This ties the commercial decision directly to statistical evidence.</div>
    </div>
</div>
''', unsafe_allow_html=True)
"""

# Extract Chart 4 and 5
chart_4_start = nb5_raw.find('    # Chart 1: Top rules by lift')
chart_5_start = nb5_raw.find('    # Chart 2: Confidence vs lift scatter')
chart_5_end = nb5_raw.find('    st.markdown("<div style=\'height:1px;')

nb5_4_chart = nb5_raw[chart_4_start:chart_5_start]
nb5_5_chart = nb5_raw[chart_5_start:chart_5_end]

nb5_6 = """
    st.markdown('''
<div id="nb5-6" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">6) Rule robustness check</h2></div>
<div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:24px;'>
    <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Robustness validation</div>
    <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Each segment's rules are validated on an 80/20 train/test split. Segments with many matched rules and low mean lift difference between train and test have stable co-purchase patterns. Segments with few matched rules should be interpreted with caution.</div>
</div>
''', unsafe_allow_html=True)
"""

nb5_7 = """
    st.markdown('''
<div id="nb5-7" style="margin-top:32px; margin-bottom:24px; border-top:1px solid #e5e7eb; padding-top:24px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">7) Campaign suggestions & Creative campaign texts</h2></div>
<div style='border-left:3px solid #111827; padding-left:20px; margin-bottom:24px;'>
    <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:6px;'>Excluded recommendations: Vegetarians (cluster 0)</div>
    <div style='font-size:16px; color:#6b7280; line-height:1.8;'>Chicken, meat, and fish are excluded from recommendations for cluster 0 (Vegetarians). The Apriori rules initially suggested these items, but they contradict the segment's defining behavioural trait. Notebook 4 confirms that this segment's identity is plant-based — the exclusion ensures campaign coherence.</div>
</div>
''', unsafe_allow_html=True)
"""

targeter_start = nb5_raw.find('    try:\n        campaign_rules = load_csv_data')
targeter_end = nb5_raw.find('    render_footer()')
nb5_8 = nb5_raw[targeter_start:targeter_end]
nb5_8 = nb5_8.replace("st.markdown(f\"<div id='nb5-8'></div>\\n\\n#### Top Association Rules for {selected_segment}\", unsafe_allow_html=True)", "st.markdown(f\"<div id='nb5-8' style='margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;'><h2 style='font-size:24px; font-weight:800; color:#111827; margin:0;'>8) Final interpretation</h2></div>\\n\\n#### Top Association Rules for {selected_segment}\", unsafe_allow_html=True)")


nb5_9 = """
    st.markdown('''
<div id="nb5-9" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">9) Cupons (Campaign Creative)</h2></div>
<div style='font-size:16px; color:#374151; line-height:1.9; margin-bottom:24px;'>
Based on the basket analysis, we designed physical and digital coupons tailored to each cluster's preferences:
<ul>
  <li><b>Vegetarians</b>: Recipe-based marketing like distributing flyers with vegetarian and vegan recipes near the fresh produce section. Ex: <i>Buy a Pack of Seitan - Get a Fresh Avocado for Free</i>.</li>
  <li><b>Wellness Urbanites</b>: Premium themed collections. Ex: <i>Healthy Pack featuring champagne, Bluetooth headphones, and yoga mats</i>.</li>
  <li><b>Students</b>: Night study essentials and quick meal kits. Ex: <i>Fuel Your Study Night</i> or <i>Quick Meal Deal (pasta, veggies, oil)</i>.</li>
  <li><b>Extended Households</b>: Cross-category family bundles. Ex: <i>Family Night Bundles - Buy Ratchet & Clank with any meal kit and get a dessert free</i>.</li>
  <li><b>Tech Enthusiasts</b>: Gadgets paired with energy products. Ex: <i>Upgrade & Recharge (Device + Energy Bar)</i> or <i>Content Creator Survival Pack</i>.</li>
  <li><b>Mature Independents</b>: Comfort groupings and cross-category. Ex: <i>Tea & Me Time (Cake + Tea + Hand Cream)</i>.</li>
  <li><b>Makro Lovers</b>: Bulk buying essentials. Ex: <i>Kitchen Basics in Bulk (12L Oil, 24-pack Water, Meatballs)</i>.</li>
</ul>
</div>
<div style="display:flex; flex-wrap:wrap; gap:16px; justify-content:center;">
''', unsafe_allow_html=True)
    import os
    cupoes_dir = IMAGENS_DIR / "cupoes"
    if cupoes_dir.exists():
        for f in os.listdir(cupoes_dir):
            if f.endswith(".png"):
                st.image(str(cupoes_dir / f), width=400)
    st.markdown("</div>", unsafe_allow_html=True)
"""

new_nb5 = (
    intro_part +
    nb5_1_2 +
    nb5_3 +
    nb5_4_chart +
    nb5_5_chart +
    nb5_6 +
    nb5_7 +
    nb5_8 +
    nb5_9 +
    "\n    render_footer()\n"
)

content = content[:start_nb5] + new_nb5 + content[end_nb5:]

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("NB5 correctly ordered and point 9 added!")
