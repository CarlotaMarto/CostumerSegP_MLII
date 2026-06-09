import os

def search_report():
    dir_path = 'scratch/extracted_report'
    files = sorted(os.listdir(dir_path))
    
    sections = [
        ("Notebook 0", "EDA"),
        ("Notebook 1", "Preprocessing"),
        ("Notebook 2", "Geography"),
        ("Notebook 3", "Clustering"),
        ("Notebook 4", "Characterization"),
        ("Notebook 5", "Association Rules")
    ]
    
    for filename in files:
        if filename.endswith('.txt'):
            path = os.path.join(dir_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            found = []
            for query, label in sections:
                if query.lower() in content.lower():
                    found.append(label)
            if found:
                print(f"{filename}: Contains {', '.join(found)}")

if __name__ == "__main__":
    search_report()
