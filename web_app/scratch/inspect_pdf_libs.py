import sys

libs = ['pypdf', 'PyPDF2', 'pdfplumber', 'fitz', 'pdfminer', 'reportlab']
print("Checking PDF libraries:")
for lib in libs:
    try:
        __import__(lib)
        print(f"  {lib}: Available")
    except ImportError:
        print(f"  {lib}: Not available")
