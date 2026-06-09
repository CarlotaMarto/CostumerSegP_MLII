import sys
import re

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# Fix CSS padding
target_css = """    backdrop-filter: none !important;
}
section[role="main"] {"""

replacement_css = """    backdrop-filter: none !important;
    padding-bottom: 0rem !important;
}
section[role="main"] {"""

if target_css in content:
    content = content.replace(target_css, replacement_css)
    print("CSS padding fixed.")
else:
    print("Warning: CSS target not found.")

# Fix Notebook 1 section 1
target_nb1 = """      <div id="nb1-1"></div>
      <!-- Stage 2: Data Cleaning -->"""

replacement_nb1 = """      <div id="nb1-1" style='border-top:1px solid #e5e7eb; padding-top:24px; margin-bottom:20px;'>
        <div style='font-size:18px; font-weight:700; color:#111827; margin-bottom:10px;'>1. Imports and Data Loading</div>
        <p style='font-size:16px; color:#374151; line-height:1.9; margin:0 0 16px 0;'>
          The dataset is loaded and prepared for cleaning. Initial inspection of data types and general structure sets the foundation for the preprocessing pipeline.
        </p>
      </div>
      <!-- Stage 2: Data Cleaning -->"""

if target_nb1 in content:
    content = content.replace(target_nb1, replacement_nb1)
    print("Notebook 1 section 1 fixed.")
else:
    print("Warning: Notebook 1 target not found.")

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
