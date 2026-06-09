import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

replacements = {
    "<div style='font-size:18px; font-weight:700; color:#111827;'>Spend profile heatmap — average lifetime spend per cluster (€)</div>":
    "<div id='nb4-3' style='font-size:18px; font-weight:700; color:#111827;'>3) Spend profile heatmap — average lifetime spend per cluster (€)</div>",
    
    "<div style='font-size:18px; font-weight:700; color:#111827;'>Behavioural and demographic profile heatmap — z-scores by cluster</div>":
    "<div id='nb4-4' style='font-size:18px; font-weight:700; color:#111827;'>4) Behavioural and demographic profile heatmap — z-scores by cluster</div>",
    
    "<div style='font-size:18px; font-weight:700; color:#111827;'>Individual radar profiles — all 8 clusters (9-axis spider chart)</div>":
    "<div id='nb4-7-1' style='font-size:18px; font-weight:700; color:#111827;'>7) Individual radar profiles — all 8 clusters (9-axis spider chart)</div>",
    
    "<div style='font-size:18px; font-weight:700; color:#111827;'>Combined radar — all 8 clusters overlaid</div>":
    "<div id='nb4-7-2' style='font-size:18px; font-weight:700; color:#111827;'>7) Combined radar — all 8 clusters overlaid</div>",
    
    "<div style='font-size:18px; font-weight:700; color:#111827;'>Average spend per category by cluster — grouped bar charts</div>":
    "<div id='nb4-7-3' style='font-size:18px; font-weight:700; color:#111827;'>7) Average spend per category by cluster — grouped bar charts</div>",
    
    "<div style='font-size:18px; font-weight:700; color:#111827;'>Key variable distributions by cluster — boxplot grid</div>":
    "<div id='nb4-7-4' style='font-size:18px; font-weight:700; color:#111827;'>7) Key variable distributions by cluster — boxplot grid</div>",
    
    "<div style='font-size:18px; font-weight:700; color:#111827;'>Geographic distribution by segment (customers per store)</div>":
    "<div id='nb4-9' style='font-size:18px; font-weight:700; color:#111827;'>9) Geographic check — Geographic distribution by segment (customers per store)</div>"
}

for old, new in replacements.items():
    content = content.replace(old, new)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Updated app.py UI titles!")
