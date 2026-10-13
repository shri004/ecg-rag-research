import json
import os


INPUT_PATH = "results/module2/retrieval_results.json"
OUTPUT_PATH = "results/module2/retrieval_evaluation.json"


# Manual relevance judgments based on the retrieved evidence.
# 1 = relevant, 0 = not relevant.
#
# Each list corresponds to Result 1, Result 2, Result 3.
RELEVANCE = {
    "MI": [1, 1, 0],
    "STTC": [1, 1, 1],
    "CD": [1, 1, 0],
    "HYP": [1, 0, 0],
    "NORM": [0, 0, 0],
}


def main():
    print("=" * 60)
    print("ECG RETRIEVAL EVALUATION")
    print("=" * 60)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        retrieval_results = json.load(f)

    evaluation = {}

    for query_label, results in retrieval_results.items():
        judgments = RELEVANCE[query_label]

        evaluated_results = []

        for result, relevance in zip(results, judgments):
            evaluated_results.append(
                {
                    "rank": result["rank"],
                    "source": result["source"],
                    "page": result["page"],
                    "superclass": result["superclass"],
                    "distance": result["distance"],
                    "relevant": relevance,
                }
            )

        relevant_count = sum(judgments)
        precision_at_3 = relevant_count / 3

        evaluation[query_label] = {
            "query": results[0]["query"],
            "top_3_relevant": relevant_count,
            "precision_at_3": precision_at_3,
            "results": evaluated_results,
        }

    total_relevant = sum(
        item["top_3_relevant"] for item in evaluation.values()
    )

    total_results = len(evaluation) * 3

    overall_precision_at_3 = total_relevant / total_results

    evaluation["overall"] = {
        "queries": len(evaluation),
        "retrieved_results": total_results,
        "relevant_results": total_relevant,
        "precision_at_3": overall_precision_at_3,
    }

    os.makedirs("results/module2", exist_ok=True)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2, ensure_ascii=False)

    print("\nResults:")
    for query_label, item in evaluation.items():
        if query_label == "overall":
            continue

        print(
            f"{query_label}: "
            f"{item['top_3_relevant']}/3 relevant | "
            f"Precision@3 = {item['precision_at_3']:.2f}"
        )

    print("\n" + "-" * 60)
    print(
        f"Overall Precision@3: "
        f"{overall_precision_at_3:.2f}"
    )

    print("\nEvaluation saved to:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()