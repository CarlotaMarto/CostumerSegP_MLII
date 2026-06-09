import os

def summarize():
    dir_path = 'scratch/extracted_report'
    files = sorted(os.listdir(dir_path))
    
    print("Report Summary by Page:")
    for filename in files:
        if filename.endswith('.txt'):
            path = os.path.join(dir_path, filename)
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Get first 3 non-empty lines
            non_empty = [l.strip() for l in lines if l.strip()]
            preview = non_empty[:3]
            preview_str = " | ".join(preview)
            print(f"{filename}: {preview_str[:120]}")

if __name__ == "__main__":
    summarize()
