import re
import os

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

start_nb4 = content.find('elif selected_page == "NB4 Characterisation":')
end_nb4 = content.find('elif selected_page == "NB5 Association Rules":')
nb4_raw = content[start_nb4:end_nb4]

nb4_parts = {}

nb4_parts['header'] = nb4_raw[:nb4_raw.find('    # Cluster sizes')]
nb4_parts['2'] = nb4_raw[nb4_raw.find('    # Cluster sizes'):nb4_raw.find('    # Interactive spend heatmap')]
nb4_parts['6_spend'] = nb4_raw[nb4_raw.find('    # Interactive spend heatmap'):nb4_raw.find('    # Interactive behavioural heatmap')]
nb4_parts['6_behav'] = nb4_raw[nb4_raw.find('    # Interactive behavioural heatmap'):nb4_raw.find('    # Community cards')]
nb4_parts['8'] = nb4_raw[nb4_raw.find('    # Community cards'):nb4_raw.find('    st.markdown("""\n<div style=\'margin-top:48px; margin-bottom:6px;\'>\n  <div style=\'font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;\'>Notebook 4 — Segment profiling charts</div>')]
nb4_parts['deep_dive'] = nb4_raw[nb4_raw.find('    st.markdown("""\n<div style=\'margin-top:48px; margin-bottom:6px;\'>\n  <div style=\'font-size:11px; font-weight:700; letter-spacing:0.12em; text-transform:uppercase; color:#9ca3af; margin-bottom:6px;\'>Notebook 4 — Segment profiling charts</div>'):nb4_raw.find('    # Spend profile heatmap (NB4 version)')]
nb4_parts['3'] = nb4_raw[nb4_raw.find('    # Spend profile heatmap (NB4 version)'):nb4_raw.find('    # Behavioural + demographic heatmap (NB4)')]
nb4_parts['4'] = nb4_raw[nb4_raw.find('    # Behavioural + demographic heatmap (NB4)'):nb4_raw.find('    # Individual radar profiles')]
nb4_parts['7_1'] = nb4_raw[nb4_raw.find('    # Individual radar profiles'):nb4_raw.find('    # Combined radar')]
nb4_parts['7_2'] = nb4_raw[nb4_raw.find('    # Combined radar'):nb4_raw.find('    # Feature barplots')]
nb4_parts['7_3'] = nb4_raw[nb4_raw.find('    # Feature barplots'):nb4_raw.find('    # Boxplot grid')]
nb4_parts['7_4'] = nb4_raw[nb4_raw.find('    # Boxplot grid'):nb4_raw.find('    # Geographic scatter by cluster (NB4)')]

s1 = '    # Geographic scatter by cluster (NB4)'
s2 = '    st.markdown("<h3 style=\'margin-top: 48px; margin-bottom: 24px;\'>Explore Clustered Data</h3>", unsafe_allow_html=True)'

nb4_parts['9'] = nb4_raw[nb4_raw.find(s1):nb4_raw.find(s2)]
nb4_parts['10'] = nb4_raw[nb4_raw.find(s2):]

nb4_parts['8'] = nb4_parts['8'].replace(
    "<div style='font-size:20px; font-weight:700; color:#111827; margin-top:40px; margin-bottom:4px;'>Your 8 customer communities</div>",
    '<div id="nb4-8" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">8) Cluster interpretation - Your 8 customer communities</h2></div>'
)

nb4_parts['6_spend'] = nb4_parts['6_spend'].replace(
    "<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>\n  <div style='font-size:18px; font-weight:700; color:#111827;'>Normalised spend profile per cluster — interactive heatmap</div>\n</div>",
    '<div id="nb4-6" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">6) Normalised comparison</h2></div>\n<div style=\'font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;\'>6.1) Normalised spend profile per cluster — interactive heatmap</div>'
)

nb4_parts['6_behav'] = nb4_parts['6_behav'].replace(
    "<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>\n  <div style='font-size:18px; font-weight:700; color:#111827;'>Normalised behavioural profile per cluster — interactive heatmap</div>\n</div>",
    '<div style="margin-top:36px; margin-bottom:12px; padding-top:24px;">\n<div style=\'font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;\'>6.2) Normalised behavioural profile per cluster — interactive heatmap</div>\n</div>'
)

nb4_parts['10'] = nb4_parts['10'].replace(
    "<h3 style='margin-top: 48px; margin-bottom: 24px;'>Explore Clustered Data</h3>",
    '<div id="nb4-10" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">10) Final segment names & export</h2></div><div style=\'font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;\'>Explore Clustered Data</div>'
)

nb4_5 = '''
    # 5) Loyalty & metadata checks
    st.markdown("""
<div id="nb4-5" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">5) Loyalty & metadata checks</h2></div>
<div style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:20px 24px; margin-top:8px; margin-bottom:32px;'>
  <p style='font-size:16px; color:#374151; line-height:1.9; margin:0;'>Loyalty program participation varies substantially by segment. Cluster 5 (Families) and Cluster 1 (Wellness Urbanites) show the highest share of loyalty card holders, suggesting strong brand attachment and habitual shopping patterns. Conversely, Cluster 4 (Tech Enthusiasts) and Cluster 2 (Students) demonstrate lower participation rates, reinforcing their opportunistic or purpose-driven behavior. Gender distribution appears relatively balanced across all clusters, confirming that segmentation was driven more by behavioral and transactional variables than by demographic traits.</p>
</div>
""", unsafe_allow_html=True)
'''

new_nb4 = (
    nb4_parts['header'] +
    nb4_parts['2'] +
    nb4_parts['deep_dive'] +
    nb4_parts['3'] +
    nb4_parts['4'] +
    nb4_5 +
    nb4_parts['6_spend'] +
    nb4_parts['6_behav'] +
    nb4_parts['7_1'] +
    nb4_parts['7_2'] +
    nb4_parts['7_3'] +
    nb4_parts['7_4'] +
    nb4_parts['8'] +
    nb4_parts['9'] +
    nb4_parts['10']
)

content = content[:start_nb4] + new_nb4 + content[end_nb4:]

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("NB4 perfectly reordered and numbered!")
