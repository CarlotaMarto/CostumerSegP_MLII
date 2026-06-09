import sys

app_path = r"c:\Users\carlo\Documents\Semestre 4\ML 2\costumer_seg_app_cm\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

start_idx = content.find('elif selected_page == "NB3 Clustering":')
end_idx = content.find('elif selected_page == "NB4 Characterisation":')

nb3_content = content[start_idx:end_idx]
with open('nb3_old.txt', 'w', encoding='utf-8') as f:
    f.write(nb3_content)
print(f'Length extracted: {len(nb3_content)}')
