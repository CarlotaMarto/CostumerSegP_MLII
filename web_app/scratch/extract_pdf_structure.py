import os
from pypdf import PdfReader

def extract_pdf():
    reader = PdfReader('report.pdf')
    num_pages = len(reader.pages)
    print(f"Total pages: {num_pages}")
    
    os.makedirs('scratch/extracted_report', exist_ok=True)
    
    # Extract outlines (Table of Contents) if any
    try:
        outlines = reader.outline
        if outlines:
            print("Outlines found:")
            def print_outline(outline, depth=0):
                for item in outline:
                    if isinstance(item, list):
                        print_outline(item, depth + 1)
                    else:
                        print("  " * depth + str(item.title))
            print_outline(outlines)
        else:
            print("No outlines found.")
    except Exception as e:
        print(f"Could not extract outline: {e}")
        
    # Extract page contents
    for i in range(num_pages):
        page = reader.pages[i]
        text = page.extract_text()
        output_path = f"scratch/extracted_report/page_{i+1:02d}.txt"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Saved page {i+1} to {output_path} ({len(text)} chars)")

if __name__ == "__main__":
    extract_pdf()
