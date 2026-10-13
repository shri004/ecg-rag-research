
import pdfplumber

CHECKS = {
    "aha_part5_HYP.pdf": [2, 3, 4, 5, 6],
    "ecg-book-general.pdf": [15, 16, 17, 136],
}

PDF_DIR = "kb/raw_guidelines"

for filename, page_numbers in CHECKS.items():
    path = f"{PDF_DIR}/{filename}"

    print("\n" + "=" * 75)
    print(f"FILE: {filename}")

    with pdfplumber.open(path) as pdf:
        for page_number in page_numbers:
            if page_number > len(pdf.pages):
                continue

            page = pdf.pages[page_number - 1]
            text = page.extract_text() or ""

            print("\n" + "-" * 60)
            print(f"PAGE {page_number}")
            print(text[:2500] if text else "[NO TEXT EXTRACTED]")
