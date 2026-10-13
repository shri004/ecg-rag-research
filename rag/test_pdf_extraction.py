import os
import pdfplumber

PDF_DIR = "kb/raw_guidelines"

for filename in os.listdir(PDF_DIR):
    if not filename.lower().endswith(".pdf"):
        continue

    path = os.path.join(PDF_DIR, filename)

    print(f"\n--- {filename} ---")

    try:
        with pdfplumber.open(path) as pdf:
            print("Pages:", len(pdf.pages))

            total_chars = 0
            pages_with_text = 0

            for page in pdf.pages:
                text = page.extract_text() or ""

                if text.strip():
                    pages_with_text += 1
                    total_chars += len(text)

            print("Pages with text:", pages_with_text)
            print("Extracted characters:", total_chars)

    except Exception as e:
        print("ERROR:", e)