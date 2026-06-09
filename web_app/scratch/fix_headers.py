import re

with open("../app.py", "r", encoding="utf-8") as f:
    content = f.read()

def repl(m):
    id_attr = m.group(1)
    title_text = m.group(2)
    return f'<div id="{id_attr}" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">{title_text}</h2></div>'

content = re.sub(r'<div id="([^"]+)"[^>]*>\s*<h3[^>]*>(.*?)</h3>\s*</div>', repl, content)

def repl2(m):
    id_attr = m.group(1)
    title_text = m.group(2)
    return f'<div id="{id_attr}" style="margin-top:64px; margin-bottom:24px; border-top:2px solid #e5e7eb; padding-top:32px;"><h2 style="font-size:24px; font-weight:800; color:#111827; margin:0;">{title_text}</h2></div>'

content = re.sub(r'<div id="([^"]+)" style=\'margin-top:36px; margin-bottom:12px; border-top:1px solid #e5e7eb; padding-top:24px;\'>\s*<div style=\'font-size:13px; font-weight:700; color:#111827;\'>([^<]+)</div>\s*</div>', repl2, content)

with open("../app.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Headers updated.")
