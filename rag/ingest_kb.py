"""
Module 2: Ingest ECG medical knowledge sources into Chroma.

Pipeline:
PDFs → page-aware extraction → chunks → embeddings → Chroma

Each chunk retains:
- source filename
- page number
- PTB-XL superclass relevance
- chunk index
"""

import argparse
import os
import re
import shutil

import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer


SUPERCLASS_HINTS = {
    "CD": "CD",
    "STTC": "STTC",
    "HYP": "HYP",
    "MI": "MI",
}


def guess_superclass(filename: str) -> str:
    """
    Assign the relevant PTB-XL diagnostic superclass based on filename.
    The textbook is treated as GENERAL because it covers multiple topics.
    """
    upper_name = filename.upper()

    for key, value in SUPERCLASS_HINTS.items():
        if key in upper_name:
            return value

    return "GENERAL"


def extract_pdf_pages(pdf_path: str) -> list:
    """
    Extract text page-by-page.

    Returns:
        [
            {"page": 1, "text": "..."},
            {"page": 2, "text": "..."},
            ...
        ]
    """
    pages = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            if text.strip():
                pages.append({
                    "page": page_number,
                    "text": text.strip()
                })

    return pages


def clean_text(text: str) -> str:
    """
    Basic whitespace cleanup while preserving the textual content.
    """
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_page_text(
    text: str,
    page_number: int,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list:
    """
    Create word-based chunks from a single page.

    Chunks do not cross page boundaries, which preserves page-level
    source traceability.
    """
    words = re.split(r"\s+", text)

    chunks = []
    start = 0
    chunk_index = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunk = " ".join(chunk_words).strip()

        if chunk:
            chunks.append({
                "page": page_number,
                "chunk_index": chunk_index,
                "text": chunk,
            })

        chunk_index += 1
        start += chunk_size - overlap

    return chunks


def main(pdf_dir: str, db_dir: str, chunk_size: int, overlap: int):
    print("=" * 60)
    print("ECG KNOWLEDGE BASE INGESTION")
    print("=" * 60)

    print("\nLoading embedding model:")
    print("  all-MiniLM-L6-v2")

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Rebuild the collection intentionally so repeated runs do not
    # accumulate duplicate/old chunks.
    if os.path.exists(db_dir):
        print(f"\nRemoving existing Chroma database: {db_dir}")
        shutil.rmtree(db_dir)

    client = chromadb.PersistentClient(path=db_dir)

    collection = client.create_collection(
        name="ecg_kb"
    )

    pdf_files = sorted(
        f for f in os.listdir(pdf_dir)
        if f.lower().endswith(".pdf")
    )

    print(f"\nFound {len(pdf_files)} PDFs:")
    for filename in pdf_files:
        print(f"  - {filename}")

    total_chunks = 0

    for filename in pdf_files:
        pdf_path = os.path.join(pdf_dir, filename)
        superclass = guess_superclass(filename)

        print("\n" + "-" * 60)
        print(f"Processing: {filename}")
        print(f"Superclass: {superclass}")

        pages = extract_pdf_pages(pdf_path)

        print(f"Pages with extracted text: {len(pages)}")

        document_chunks = []

        for page_data in pages:
            cleaned = clean_text(page_data["text"])

            page_chunks = chunk_page_text(
                cleaned,
                page_data["page"],
                chunk_size,
                overlap,
            )

            document_chunks.extend(page_chunks)

        print(f"Chunks created: {len(document_chunks)}")

        if not document_chunks:
            print("WARNING: No chunks created. Skipping.")
            continue

        texts = [
            item["text"]
            for item in document_chunks
        ]

        embeddings = embedder.encode(
            texts,
            show_progress_bar=False,
        ).tolist()

        ids = []
        metadatas = []

        for item in document_chunks:
            chunk_id = (
                f"{filename}"
                f"_page_{item['page']}"
                f"_chunk_{item['chunk_index']}"
            )

            ids.append(chunk_id)

            metadatas.append({
                "source": filename,
                "page": item["page"],
                "superclass": superclass,
                "chunk_index": item["chunk_index"],
            })

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        total_chunks += len(document_chunks)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total chunks stored: {total_chunks}")
    print(f"Chroma collection count: {collection.count()}")
    print(f"Database location: {db_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pdf-dir",
        default="kb/raw_guidelines",
    )

    parser.add_argument(
        "--db-dir",
        default="vectordb",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
    )

    args = parser.parse_args()

    main(
        args.pdf_dir,
        args.db_dir,
        args.chunk_size,
        args.overlap,
    )