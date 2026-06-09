app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix the quotes!
content = content.replace(
    'st.markdown("<div id="nb4-8" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">8) Cluster interpretation - Your 8 customer communities</h2></div>", unsafe_allow_html=True)',
    'st.markdown(\'<div id="nb4-8" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">8) Cluster interpretation - Your 8 customer communities</h2></div>\', unsafe_allow_html=True)'
)

content = content.replace(
    'st.markdown("<div id="nb4-6" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">6) Normalised comparison</h2></div>\\n<div style=\'font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;\'>6.1) Normalised spend profile per cluster — interactive heatmap</div>", unsafe_allow_html=True)',
    'st.markdown("""<div id="nb4-6" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">6) Normalised comparison</h2></div>\\n<div style=\'font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;\'>6.1) Normalised spend profile per cluster — interactive heatmap</div>""", unsafe_allow_html=True)'
)

content = content.replace(
    'st.markdown("<div id="nb4-10" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">10) Final segment names & export</h2></div><div style=\'font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;\'>Explore Clustered Data</div>", unsafe_allow_html=True)',
    'st.markdown("""<div id="nb4-10" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">10) Final segment names & export</h2></div><div style=\'font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;\'>Explore Clustered Data</div>""", unsafe_allow_html=True)'
)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
