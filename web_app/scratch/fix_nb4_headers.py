import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
    """<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div id='nb4-3' style='font-size:18px; font-weight:700; color:#111827;'>3) Spend profile heatmap — average lifetime spend per cluster (€)</div>
</div>""":
    """<div id="nb4-3" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">3) Spend profile heatmap</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Average lifetime spend per cluster (€)</div>""",

    """<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div id='nb4-4' style='font-size:18px; font-weight:700; color:#111827;'>4) Behavioural and demographic profile heatmap — z-scores by cluster</div>
</div>""":
    """<div id="nb4-4" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">4) Behavioural & demographic profile</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Z-scores by cluster</div>""",

    """<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div id='nb4-7-1' style='font-size:18px; font-weight:700; color:#111827;'>7) Individual radar profiles — all 8 clusters (9-axis spider chart)</div>
</div>""":
    """<div id="nb4-7" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">7) Feature plots</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>7.1) Individual radar profiles — all 8 clusters (9-axis spider chart)</div>""",

    """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Combined radar — all 8 clusters overlaid</div>
</div>""":
    """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.2) Combined radar — all 8 clusters overlaid</div>
</div>""",

    """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Average spend per category by cluster — grouped bar charts</div>
</div>""":
    """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.3) Average spend per category by cluster — grouped bar charts</div>
</div>""",

    """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Key variable distributions by cluster — boxplot grid</div>
</div>""":
    """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.4) Key variable distributions by cluster — boxplot grid</div>
</div>""",

    """<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>Geographic distribution by cluster — static scatter</div>
</div>""":
    """<div id="nb4-9" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">9) Geographic check</h2></div>
<div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:12px;'>Geographic distribution by cluster — static scatter</div>"""
}

# Add fallback replacements just in case `fix_app_ui_4` hadn't correctly updated some elements
replacements["""<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div id='nb4-7-2' style='font-size:18px; font-weight:700; color:#111827;'>7) Combined radar — all 8 clusters overlaid</div>
</div>"""] = """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.2) Combined radar — all 8 clusters overlaid</div>
</div>"""

replacements["""<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div id='nb4-7-3' style='font-size:18px; font-weight:700; color:#111827;'>7) Average spend per category by cluster — grouped bar charts</div>
</div>"""] = """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.3) Average spend per category by cluster — grouped bar charts</div>
</div>"""

replacements["""<div style='margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;'>
  <div id='nb4-7-4' style='font-size:18px; font-weight:700; color:#111827;'>7) Key variable distributions by cluster — boxplot grid</div>
</div>"""] = """<div style='margin-top:36px; margin-bottom:12px; padding-top:24px;'>
  <div style='font-size:18px; font-weight:700; color:#111827;'>7.4) Key variable distributions by cluster — boxplot grid</div>
</div>"""

for old, new in replacements.items():
    content = content.replace(old, new)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated app.py UI headers!")
