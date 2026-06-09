import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Fix the scroll script
old_scroll = """        <script>
            var body = window.parent.document.querySelector(".main");
            if (body) {
                body.scrollTo(0, 0);
            }
            window.parent.scrollTo(0, 0);
        </script>"""

new_scroll = """        <script>
            setTimeout(function() {
                var body = window.parent.document.querySelector(".main");
                if (body) {
                    body.scrollTo({top: 0, behavior: 'instant'});
                }
                window.parent.scrollTo({top: 0, behavior: 'instant'});
            }, 100);
        </script>"""
content = content.replace(old_scroll, new_scroll)

# 2. Fix NB3 topics
# We want to replace the whole content from `<div id="nb3-1"></div><div id="nb3-2"></div><div id="nb3-3"></div>`
# down to the end of nb3 with a properly ordered version.

nb3_start_marker = """      <div id="nb3-1"></div><div id="nb3-2"></div><div id="nb3-3"></div>"""

nb3_new_content = """      <div id="nb3-1" style='background:#f9fafb; border:1px solid #e5e7eb; border-radius:12px; padding:16px 20px; margin-bottom:28px;'>
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
"""

# The rest of the content needs to be ordered:
# We will just inject the missing elements and reorder the existing ones by manipulating the HTML strings.
# But it's easier to just find `    # ---- NB3 Charts ----` and put the text parts before it.
# Actually, the user wants the charts interspersed with the text just like in the notebook, OR they just want the numbers ordered.
# Let's extract the chart blocks and put them in order!
