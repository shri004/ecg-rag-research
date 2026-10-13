"""
Retrieval diagnostic for Module 2. Run BEFORE changing anything.

For each class it answers three separate questions:
  1. What do the top-N chunks actually look like (source/page/tag + text)?
  2. How much of the top-N comes from the GENERAL textbook vs class-tagged
     AHA chunks? (the KB is ~76% textbook, so dominance is a hypothesis to test)
  3. For each domain keyword you expect in a good answer: does ANY chunk in
     the whole KB contain it, and if so at what rank does the first such
     chunk appear for this query?
        - keyword in no chunk            -> coverage / extraction / chunking problem
        - keyword in chunk(s), rank > N  -> ranking / query / embedding problem

Usage:
    python rag/diagnose_retrieval.py --db-dir vectordb --collection <name> \
        --spec results/module2/diagnostic_spec.json --top-n 10

diagnostic_spec.json format (edit freely -- these are YOUR queries/keywords):
{
  "HYP": {"queries": ["criteria for left ventricular hypertrophy on ECG"],
          "keywords": ["Sokolow", "Cornell", "voltage"]},
  "NORM": {"queries": ["normal ECG intervals PR QRS QT and axis"],
           "keywords": ["PR interval", "QRS duration"]}
}
"""
import argparse
import json
from collections import Counter


def first_rank_with_keyword(ranked_docs, keyword):
    kw = keyword.lower()
    for i, doc in enumerate(ranked_docs, start=1):
        if kw in doc.lower():
            return i
    return None


def diagnose(collection, embed_fn, spec, top_n=10, snippet_chars=160):
    total = collection.count()
    all_docs = collection.get(include=["documents", "metadatas"])
    report = {}
    for cls, item in spec.items():
        keywords = item.get("keywords", [])
        # keyword coverage over the WHOLE KB (independent of any query)
        coverage = {}
        for kw in keywords:
            hits = [m for d, m in zip(all_docs["documents"], all_docs["metadatas"])
                    if kw.lower() in d.lower()]
            coverage[kw] = {
                "n_chunks_containing": len(hits),
                "sources_pages": sorted({(m.get("source"), m.get("page")) for m in hits},
                                        key=lambda x: (str(x[0]), str(x[1])))[:8],
            }
        report[cls] = {"coverage": coverage, "queries": []}

        for q in item["queries"]:
            res = collection.query(query_embeddings=embed_fn([q]), n_results=total,
                                   include=["documents", "metadatas", "distances"])
            docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
            top = list(zip(docs[:top_n], metas[:top_n], dists[:top_n]))
            tag_mix = Counter(m.get("superclass", "?") for _, m, _ in top)
            kw_ranks = {kw: first_rank_with_keyword(docs, kw) for kw in keywords}
            report[cls]["queries"].append({
                "query": q,
                "top_n_tag_mix": dict(tag_mix),
                "first_rank_of_keyword_in_full_ranking": kw_ranks,
                "top": [{"rank": i + 1, "source": m.get("source"), "page": m.get("page"),
                         "tag": m.get("superclass"), "distance": round(float(d), 4),
                         "snippet": doc[:snippet_chars].replace("\n", " ")}
                        for i, (doc, m, d) in enumerate(top)],
            })
    return report


def print_report(report):
    for cls, r in report.items():
        print("=" * 78)
        print(f"CLASS {cls}")
        print("-- keyword coverage across whole KB --")
        for kw, c in r["coverage"].items():
            print(f"  '{kw}': in {c['n_chunks_containing']} chunk(s) {c['sources_pages']}")
        for qr in r["queries"]:
            print(f"\n  QUERY: {qr['query']}")
            print(f"  top-N tag mix: {qr['top_n_tag_mix']}")
            print(f"  first rank containing keyword: {qr['first_rank_of_keyword_in_full_ranking']}")
            for t in qr["top"]:
                print(f"   #{t['rank']:>2} [{t['tag']}] {t['source']} p.{t['page']} "
                      f"d={t['distance']}  {t['snippet']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-dir", default="vectordb")
    ap.add_argument("--collection", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--out", default="results/module2/diagnostic_report.json")
    args = ap.parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    embed_fn = lambda texts: model.encode(texts).tolist()
    col = chromadb.PersistentClient(path=args.db_dir).get_collection(args.collection)
    with open(args.spec) as f:
        spec = json.load(f)

    rep = diagnose(col, embed_fn, spec, args.top_n)
    print_report(rep)
    with open(args.out, "w") as f:
        json.dump(rep, f, indent=2, default=str)
    print(f"\nSaved {args.out}")