
import json
import os

import chromadb
from sentence_transformers import SentenceTransformer


DB_DIR = "vectordb"
COLLECTION_NAME = "ecg_kb"

TOP_K = 3
CANDIDATES_PER_TAG = 10

QUERIES = {
    "MI": "What ECG findings are associated with myocardial infarction?",
    "STTC": "What ECG changes are associated with ST segment and T wave abnormalities?",
    "CD": "What ECG findings indicate conduction disturbances?",
    "HYP": "What ECG findings are associated with cardiac hypertrophy?",
    "NORM": "What are the characteristics of a normal ECG?",
}


def format_result(rank, query, document, metadata, distance, chunk_id):
    return {
        "rank": rank,
        "query": query,
        "source": metadata["source"],
        "page": metadata["page"],
        "superclass": metadata["superclass"],
        "chunk_index": metadata["chunk_index"],
        "chunk_id": chunk_id,
        "distance": float(distance),
        "text": document,
    }


def global_retrieval(collection, query_embedding, query):
    """Baseline: retrieve the globally closest chunks."""
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=TOP_K,
    )

    output = []

    for rank, (document, metadata, distance, chunk_id) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ),
        start=1,
    ):
        output.append(
            format_result(
                rank, query, document, metadata, distance, chunk_id
            )
        )

    return output


def class_aware_retrieval(collection, query_embedding, query, label):
    """
    Reserve two results from the matching superclass and one from GENERAL.
    NORM retrieves three GENERAL results because no NORM-tagged source
    currently exists.
    """
    if label == "NORM":
        quotas = {"GENERAL": TOP_K}
    else:
        quotas = {label: 2, "GENERAL": 1}

    selected = {}

    for tag, quota in quotas.items():
        available = collection.get(
            where={"superclass": tag},
            include=["metadatas"],
        )

        available_count = len(available["ids"])

        if available_count == 0:
            print(f"WARNING: No chunks found for tag {tag}")
            continue

        candidate_count = min(
            max(quota, CANDIDATES_PER_TAG),
            available_count,
        )

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=candidate_count,
            where={"superclass": tag},
        )

        for document, metadata, distance, chunk_id in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
            results["ids"][0],
        ):
            selected[chunk_id] = (
                document,
                metadata,
                distance,
                chunk_id,
                tag,
            )

    # Select each tag's quota separately to enforce source diversity.
    output = []

    for tag, quota in quotas.items():
        tag_candidates = [
            item for item in selected.values() if item[4] == tag
        ]

        tag_candidates.sort(key=lambda item: item[2])

        for document, metadata, distance, chunk_id, _ in (
            tag_candidates[:quota]
        ):
            output.append(
                format_result(
                    0, query, document, metadata, distance, chunk_id
                )
            )

    # Sort selected results by distance for consistent display.
    output.sort(key=lambda item: item["distance"])

    for rank, item in enumerate(output, start=1):
        item["rank"] = rank

    return output[:TOP_K]


def print_results(title, results):
    print(f"\n{title}")

    for item in results:
        print(f"\nResult {item['rank']}")
        print(f"Source: {item['source']}")
        print(f"Page: {item['page']}")
        print(f"Superclass: {item['superclass']}")
        print(f"Distance: {item['distance']:.4f}")
        print(f"Text: {item['text'][:500]}...")


def main():
    print("=" * 60)
    print("ECG KNOWLEDGE BASE RETRIEVAL COMPARISON")
    print("=" * 60)

    print("\nLoading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nOpening Chroma database...")
    client = chromadb.PersistentClient(path=DB_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    print(f"Collection: {COLLECTION_NAME}")
    print(f"Total chunks: {collection.count()}")

    comparison = {
        "metadata": {
            "embedding_model": "all-MiniLM-L6-v2",
            "top_k": TOP_K,
            "candidates_per_tag": CANDIDATES_PER_TAG,
            "class_aware_method": (
                "Reserve two matching-superclass results and one GENERAL "
                "result for MI, STTC, CD, and HYP. NORM uses GENERAL only. "
                "Sort selected results by embedding distance."
            ),
        },
        "queries": {},
    }

    for label, query in QUERIES.items():
        print("\n" + "=" * 60)
        print(f"QUERY: {label}")
        print(query)

        query_embedding = embedder.encode([query]).tolist()

        baseline = global_retrieval(
            collection, query_embedding, query
        )

        class_aware = class_aware_retrieval(
            collection, query_embedding, query, label
        )

        comparison["queries"][label] = {
            "query": query,
            "global_baseline": baseline,
            "class_aware": class_aware,
        }

        print_results("GLOBAL BASELINE", baseline)
        print_results("CLASS-AWARE RETRIEVAL", class_aware)

    os.makedirs("results/module2", exist_ok=True)

    output_path = "results/module2/retrieval_comparison.json"

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("RETRIEVAL COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_path}")
    print("Original retrieval_results.json was not modified.")


if __name__ == "__main__":
    main()
