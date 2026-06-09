import re

with open("app.py", "r", encoding="utf-8") as f:
    content = f.read()

nb4_content = ""
if 'elif selected_page == "NB4 Characterisation":' in content:
    parts = content.split('elif selected_page == "NB4 Characterisation":')
    if 'elif selected_page == "Targeter Promotion":' in parts[1]:
        nb4_content = parts[1].split('elif selected_page == "Targeter Promotion":')[0]
    else:
        nb4_content = parts[1]

headers = re.findall(r'<h[234].*?>(.*?)</h[234]>|<div.*?font-size:(?:18|20|24)px.*?(?:font-weight:700|font-weight:800).*?>(.*?)</div>', nb4_content)
for h in headers:
    print(h[0] if h[0] else h[1])
