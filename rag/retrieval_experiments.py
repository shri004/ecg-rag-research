#!/usr/bin/env python3
"""
Module 2 retrieval experiments -- controlled, reuses the EXISTING Chroma DB.

NOTHING here writes to the vector DB, to your baseline files, or to your
existing manual judgments. It only reads the DB and writes new files.

Commands (run from the project root):

  audit             Flag non-content chunks (title pages, reference lists,
                    manual page exclusions). Writes an audit CSV for you to
                    spot-check BEFORE trusting the filter.
  run               Run retrieval arms on the frozen queries; write results
                    (IDs/pages only, no textbook text) + a blind judging sheet.
  merge-judgments   Fold a filled judging sheet into eval/retrieval/judgments.csv
                    (text is dropped, so that file is safe to commit).
  score             Compute metrics per arm/class + paired bootstrap.
  agreement         Exact agreement + Cohen's kappa between two judges.

Arms: dense | dense_clean | bm25_clean | hybrid_clean
  dense         baseline: dense retrieval over ALL chunks
  dense_clean   dense retrieval after removing non-content chunks
  bm25_clean    lexical BM25 over clean chunks (ablation)
  hybrid_clean  reciprocal-rank fusion of dense_clean + bm25_clean

The non-content filter is: (a) pages you list in kb/exclude_pages.json,
(b) a citation-density heuristic (reference lists), (c) a short marker list.
Thresholds below are OUR CHOICES, untested on your real chunks -- that is
why `audit` exists. Review its output before running anything else.
"""
import argparse
import csv
import datetime
import hashlib
import json
import math
import os
import random
import re
import subprocess
import sys
from collections import Counter, defaultdict

QUERIES_HELDOUT = "eval/retrieval/queries_heldout.json"
QUERIES_DEV = "eval/retrieval/queries_dev.json"      # optional
JUDGMENTS = "eval/retrieval/judgments.csv"           # committed, NO text
WORK_DIR = "eval/retrieval/work"                     # gitignored, has text
EXCLUDE = "kb/exclude_pages.json"
RUNS_DIR = "results/module2/runs"
SUMMARY_DIR = "results/module2"

RRF_K = 60
CITE_THRESH = 3.0           # citation-like hits per 100 words => reference list
MAX_FLAG_FRAC = 0.6         # safety valve: abort if a source is mostly flagged
DEFAULT_MARKERS = ["key words", "writing group disclosures", "all rights reserved", "isbn"]
ALL_ARMS = ["dense", "dense_clean", "bm25_clean", "hybrid_clean"]

CITE_RE = re.compile(r"\b(?:19|20)\d{2}\s*;\s*\d{1,3}\s*:\s*[A-Za-z]?\d{1,5}")
ETAL_RE = re.compile(r"\bet\s+al\b", re.I)
STOP = set("the of and a an in on for to with by is are as at or from that this be it its".split())


# ----------------------------------------------------------------- loading --
def load_collection(db_dir, name):
    import chromadb
    client = chromadb.PersistentClient(path=db_dir)
    try:
        return client.get_collection(name)
    except Exception:
        names = [getattr(c, "name", c) for c in client.list_collections()]
        sys.exit(f"Collection '{name}' not found in {db_dir}. Available: {names}")


def load_chunks(collection):
    d = collection.get(include=["documents", "metadatas"])
    chunks = []
    for cid, doc, meta in zip(d["ids"], d["documents"], d["metadatas"]):
        meta = meta or {}
        chunks.append({"id": cid, "text": doc or "", "source": meta.get("source"),
                       "page": meta.get("page"), "tag": meta.get("superclass")})
    chunks.sort(key=lambda c: c["id"])          # deterministic order
    return chunks


def get_embed_fn(model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return lambda texts: model.encode(texts).tolist()


def load_exclude_cfg(path=EXCLUDE):
    if not os.path.exists(path):
        return {"pages": {}, "markers": DEFAULT_MARKERS}
    with open(path) as f:
        raw = json.load(f)
    pages = {s: set(p) for s, p in raw.items() if not s.startswith("_")}
    return {"pages": pages, "markers": raw.get("_markers", DEFAULT_MARKERS)}


def load_queries(path, set_name, required):
    if not os.path.exists(path):
        if required:
            sys.exit(f"Missing {path}. Create and commit it BEFORE running any arm.")
        return []
    with open(path) as f:
        raw = json.load(f)
    out = []
    for cls, items in raw.items():
        if cls.startswith("_"):
            continue
        for it in items:
            out.append({"set": set_name, "class": cls, "id": it["id"], "text": it["text"]})
    return out


def load_judgments(path=JUDGMENTS):
    j = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                j[(r["query_id"], r["chunk_id"])] = (int(r["grade"]), int(r.get("garbled") or 0))
    return j


# ----------------------------------------------------- non-content filter ---
def nonContent_reason(chunk, cfg):
    if chunk["page"] in cfg["pages"].get(chunk["source"], set()):
        return "manual_exclude_page"
    text = chunk["text"]
    words = max(len(text.split()), 1)
    cites = len(CITE_RE.findall(text)) + len(ETAL_RE.findall(text))
    if cites * 100.0 / words >= CITE_THRESH:
        return f"citation_density>={CITE_THRESH}"
    low = text.lower()
    for m in cfg["markers"]:
        if m in low:
            return f"marker:{m}"
    return None


def build_flags(chunks, cfg, force=False):
    flags = {c["id"]: nonContent_reason(c, cfg) for c in chunks}
    per_src = defaultdict(lambda: [0, 0])
    for c in chunks:
        per_src[c["source"]][1] += 1
        per_src[c["source"]][0] += flags[c["id"]] is not None
    bad = [(s, f, n) for s, (f, n) in per_src.items() if n and f / n > MAX_FLAG_FRAC]
    if bad and not force:
        sys.exit("SAFETY STOP: filter flags >{:.0%} of chunks in {} -- the heuristic/markers "
                 "are probably wrong for your PDFs. Inspect the audit, fix {} "
                 "(or pass --force).".format(MAX_FLAG_FRAC, bad, EXCLUDE))
    return flags, per_src


# ------------------------------------------------------------------- BM25 ---
def tok(text):
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP]


class BM25:
    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.n = len(docs_tokens)
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / self.n) if self.n else 0.0
        self.tf = [Counter(d) for d in docs_tokens]
        df = Counter()
        for d in docs_tokens:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def scores(self, q_tokens):
        out = []
        for i in range(self.n):
            s, tf, dl = 0.0, self.tf[i], self.dl[i]
            for t in set(q_tokens):
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (
                        f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0)))
            out.append(s)
        return out


# ---------------------------------------------------------------- ranking ---
def rrf(rank_lists, k=RRF_K):
    score = defaultdict(float)
    for ranking in rank_lists:
        for pos, cid in enumerate(ranking, start=1):
            score[cid] += 1.0 / (k + pos)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


def rank_all_arms(arms, query_text, collection, embed_fn, clean_chunks, bm25, flags, k):
    n_total = collection.count()
    res = collection.query(query_embeddings=embed_fn([query_text]), n_results=n_total,
                           include=["distances"])
    dense_ids, dense_d = res["ids"][0], res["distances"][0]
    clean_ids = {c["id"] for c in clean_chunks}
    dense_clean = [(i, d) for i, d in zip(dense_ids, dense_d) if i in clean_ids]

    out = {}
    if "dense" in arms:
        out["dense"] = list(zip(dense_ids, dense_d))[:k]
    if "dense_clean" in arms:
        out["dense_clean"] = dense_clean[:k]
    if {"bm25_clean", "hybrid_clean"} & set(arms):
        sc = bm25.scores(tok(query_text))
        ranked = sorted([(clean_chunks[i]["id"], s) for i, s in enumerate(sc) if s > 0],
                        key=lambda x: (-x[1], x[0]))
        if "bm25_clean" in arms:
            out["bm25_clean"] = ranked[:k]
        if "hybrid_clean" in arms:
            out["hybrid_clean"] = rrf([[i for i, _ in dense_clean], [i for i, _ in ranked]])[:k]
    return out


# ---------------------------------------------------------------- helpers ---
def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def sha256_of(path):
    if not os.path.exists(path):
        return None
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# --------------------------------------------------------------- commands ---
def cmd_audit(args):
    col = load_collection(args.db_dir, args.collection)
    chunks = load_chunks(col)
    cfg = load_exclude_cfg()
    flags, per_src = build_flags(chunks, cfg, force=args.force)

    os.makedirs(WORK_DIR, exist_ok=True)
    os.makedirs(SUMMARY_DIR, exist_ok=True)
    path = os.path.join(WORK_DIR, "boilerplate_audit.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["chunk_id", "source", "page", "flagged", "reason", "snippet"])
        for c in chunks:
            w.writerow([c["id"], c["source"], c["page"], int(flags[c["id"]] is not None),
                        flags[c["id"]] or "", c["text"][:150].replace("\n", " ")])
    summary = {s: {"flagged": f, "total": n} for s, (f, n) in per_src.items()}
    with open(os.path.join(SUMMARY_DIR, "boilerplate_summary.json"), "w") as f:
        json.dump({"per_source": summary, "exclude_sha256": sha256_of(EXCLUDE)}, f, indent=2)
    print(f"Chunks: {len(chunks)}")
    for s, v in summary.items():
        print(f"  {s}: flagged {v['flagged']}/{v['total']}")
    print(f"Audit written to {path} (gitignored: contains text).")
    print("NEXT: open it, (1) confirm flagged rows are really non-content, "
          "(2) spot-check ~20 UNflagged rows for missed junk, then edit "
          f"{EXCLUDE} if needed.")


def cmd_run(args, embed_fn=None):
    arms = [a.strip() for a in args.arms.split(",")]
    for a in arms:
        if a not in ALL_ARMS:
            sys.exit(f"Unknown arm '{a}'. Choose from {ALL_ARMS}")
    queries = load_queries(QUERIES_HELDOUT, "heldout", required=True)
    queries += load_queries(QUERIES_DEV, "dev", required=False)
    col = load_collection(args.db_dir, args.collection)
    chunks = load_chunks(col)
    id2c = {c["id"]: c for c in chunks}
    cfg = load_exclude_cfg()
    flags, _ = build_flags(chunks, cfg, force=args.force)
    clean_chunks = [c for c in chunks if flags[c["id"]] is None]
    bm25 = BM25([tok(c["text"]) for c in clean_chunks])
    embed_fn = embed_fn or get_embed_fn(args.model)

    results = []
    for q in queries:
        ranked = rank_all_arms(arms, q["text"], col, embed_fn, clean_chunks, bm25, flags, args.k)
        arm_out = {}
        for arm, items in ranked.items():
            arm_out[arm] = [{"rank": r, "id": cid, "source": id2c[cid]["source"],
                             "page": id2c[cid]["page"], "tag": id2c[cid]["tag"],
                             "score": round(float(sc), 5), "nonContent": flags[cid]}
                            for r, (cid, sc) in enumerate(items, start=1)]
        results.append({"set": q["set"], "class": q["class"], "query_id": q["id"],
                        "query": q["text"], "arms": arm_out})

    run_dir = os.path.join(RUNS_DIR, args.run_id)
    os.makedirs(run_dir, exist_ok=True)
    config = {"run_id": args.run_id, "arms": arms, "k": args.k, "pool_depth": args.pool_depth,
              "model": args.model, "collection": args.collection, "n_chunks": len(chunks),
              "n_clean_chunks": len(clean_chunks), "rrf_k": RRF_K, "cite_thresh": CITE_THRESH,
              "exclude_sha256": sha256_of(EXCLUDE), "queries_sha256": sha256_of(QUERIES_HELDOUT),
              "git_commit": git_commit(),
              "timestamp": datetime.datetime.now().isoformat(timespec="seconds")}
    with open(os.path.join(run_dir, "retrieved.json"), "w") as f:
        json.dump({"config": config, "results": results}, f, indent=2)

    # blind pooled judging sheet: union of top-`pool_depth` over arms, minus already judged
    judged = load_judgments()
    rows = []
    for r in results:
        pool = {}
        for items in r["arms"].values():
            for it in items[:args.pool_depth]:
                pool[it["id"]] = it
        for cid in pool:
            if (r["query_id"], cid) not in judged:
                rows.append({"query_id": r["query_id"], "query": r["query"], "chunk_id": cid,
                             "source": id2c[cid]["source"], "page": id2c[cid]["page"],
                             "text": id2c[cid]["text"], "grade": "", "garbled": "", "judge": ""})
    random.Random(args.run_id).shuffle(rows)
    os.makedirs(WORK_DIR, exist_ok=True)
    todo = os.path.join(WORK_DIR, f"todo_{args.run_id}.csv")
    with open(todo, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "query", "chunk_id", "source", "page",
                                          "text", "grade", "garbled", "judge"])
        w.writeheader()
        w.writerows(rows)
    print(f"Run '{args.run_id}': {len(queries)} queries, arms={arms}")
    print(f"Retrieved (IDs/pages only): {run_dir}/retrieved.json")
    print(f"{len(rows)} chunk(s) to judge -> {todo}")
    print("Judge: grade 0/1/2, garbled 0/1, judge initials. Rubric in the docs/prompt. "
          "Then run merge-judgments.")



def cmd_merge(args):
    with open(args.todo, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    existing = load_judgments()
    new, blank, conflict = [], 0, 0

    for r in rows:
        g = (r.get("grade") or "").strip()

        if g == "":
            blank += 1
            continue

        try:
            g_num = float(g)
        except ValueError:
            sys.exit(
                f"Bad grade '{g}' for {r['query_id']}/{r['chunk_id']} "
                "(use 0/1/2)."
            )

        if g_num not in {0.0, 1.0, 2.0}:
            sys.exit(
                f"Bad grade '{g}' for {r['query_id']}/{r['chunk_id']} "
                "(use 0/1/2)."
            )

        g = str(int(g_num))

        gb = (r.get("garbled") or "0").strip() or "0"

        try:
            gb_num = float(gb)
        except ValueError:
            sys.exit(
                f"Bad garbled '{gb}' for {r['query_id']}/{r['chunk_id']} "
                "(use 0/1)."
            )

        if gb_num not in {0.0, 1.0}:
            sys.exit(
                f"Bad garbled '{gb}' for {r['query_id']}/{r['chunk_id']} "
                "(use 0/1)."
            )

        gb = str(int(gb_num))
        key = (r["query_id"], r["chunk_id"])

        if key in existing:
            conflict += 1
            continue

        new.append({
            "query_id": r["query_id"],
            "chunk_id": r["chunk_id"],
            "source": r["source"],
            "page": r["page"],
            "grade": g,
            "garbled": gb,
            "judge": r.get("judge", "")
        })

        existing[key] = (int(g), int(gb))

    os.makedirs(os.path.dirname(JUDGMENTS), exist_ok=True)
    write_header = not os.path.exists(JUDGMENTS)

    with open(JUDGMENTS, "a", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "query_id", "chunk_id", "source", "page",
                "grade", "garbled", "judge"
            ]
        )

        if write_header:
            w.writeheader()

        w.writerows(new)

    print(
        f"Merged {len(new)} judgments; "
        f"skipped {blank} blank, {conflict} already-judged."
    )

def dcg(grades):
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def paired_bootstrap(diffs, n=2000, seed=0):
    rng = random.Random(seed)
    m = len(diffs)
    means = sorted(sum(rng.choice(diffs) for _ in range(m)) / m for _ in range(n))
    return sum(diffs) / m, means[int(0.025 * n)], means[int(0.975 * n) - 1]


def cmd_score(args):
    path = os.path.join(RUNS_DIR, args.run_id, "retrieved.json")
    with open(path) as f:
        data = json.load(f)
    k = data["config"]["k"]
    judged = load_judgments()
    per_query = defaultdict(dict)     # arm -> qid -> metrics
    qinfo = {}
    missing = []
    for r in data["results"]:
        if r["set"] != args.set:
            continue
        qinfo[r["query_id"]] = r["class"]
        ideal = sorted([g for (qid, _), (g, _) in judged.items() if qid == r["query_id"]],
                       reverse=True)[:k]
        for arm, items in r["arms"].items():
            grades, garbled, junk = [], [], []
            for it in items[:k]:
                key = (r["query_id"], it["id"])
                if key not in judged:
                    missing.append(key)
                    continue
                grades.append(judged[key][0])
                garbled.append(judged[key][1])
                junk.append(int(it["nonContent"] is not None))
            if len(grades) < len(items[:k]):
                continue
            n = max(len(grades), 1)
            first = next((i for i, g in enumerate(grades, 1) if g == 2), None)
            idcg = dcg(ideal)
            per_query[arm][r["query_id"]] = {
                "p_strict": sum(g == 2 for g in grades) / k,
                "p_lenient": sum(g >= 1 for g in grades) / k,
                "hit": int(any(g == 2 for g in grades)),
                "mrr": (1.0 / first) if first else 0.0,
                "ndcg": (dcg(grades) / idcg) if idcg > 0 else None,
                "junk": sum(junk) / n, "garbled": sum(garbled) / n,
                "n_returned": len(grades)}
    if missing and not args.allow_missing:
        sys.exit(f"{len(set(missing))} retrieved (query, chunk) pairs are UNJUDGED, e.g. "
                 f"{sorted(set(missing))[:3]}. Judge + merge them, or pass --allow-missing.")

    outdir = os.path.join(RUNS_DIR, args.run_id)
    summary = {}
    print(f"\nRun {args.run_id} | set={args.set} | k={k}\n")
    hdr = f"{'arm':<14}{'P@k strict':>11}{'P@k lenient':>12}{'hit@k':>8}{'MRR@k':>8}{'nDCG@k':>8}{'junk@k':>8}{'garbled@k':>10}{'nQ':>5}"
    print(hdr)
    for arm, qm in per_query.items():
        vals = list(qm.values())
        nd = [v["ndcg"] for v in vals if v["ndcg"] is not None]
        mean = lambda key: sum(v[key] for v in vals) / len(vals)
        summary[arm] = {"n_queries": len(vals), "p_strict": mean("p_strict"),
                        "p_lenient": mean("p_lenient"), "hit": mean("hit"), "mrr": mean("mrr"),
                        "ndcg": (sum(nd) / len(nd)) if nd else None, "junk": mean("junk"),
                        "garbled": mean("garbled"),
                        "by_class": {}}
        for cls in sorted(set(qinfo.values())):
            cv = [v for q, v in qm.items() if qinfo[q] == cls]
            if cv:
                summary[arm]["by_class"][cls] = {
                    "p_strict": sum(v["p_strict"] for v in cv) / len(cv),
                    "hit": sum(v["hit"] for v in cv) / len(cv), "n": len(cv)}
        s = summary[arm]
        ndtxt = f"{s['ndcg']:.2f}" if s["ndcg"] is not None else "n/a"
        print(f"{arm:<14}{s['p_strict']:>11.2f}{s['p_lenient']:>12.2f}{s['hit']:>8.2f}"
              f"{s['mrr']:>8.2f}{ndtxt:>8}{s['junk']:>8.2f}{s['garbled']:>10.2f}{s['n_queries']:>5}")
    print("\nPer-class strict P@k / hit@k:")
    for arm, s in summary.items():
        print(f"  {arm}: " + "  ".join(f"{c}={v['p_strict']:.2f}/{v['hit']:.2f}(n={v['n']})"
                                       for c, v in s["by_class"].items()))

    boot = {}
    print("\nPaired bootstrap of strict P@k difference (descriptive; small n):")
    pairs = [("dense", "dense_clean"), ("dense_clean", "hybrid_clean"),
             ("dense_clean", "bm25_clean")]
    for a, b in [(a, b) for a, b in pairs if a in per_query and b in per_query]:
        common = sorted(set(per_query[a]) & set(per_query[b]))
        if len(common) < 2:
            continue
        diffs = [per_query[b][q]["p_strict"] - per_query[a][q]["p_strict"] for q in common]
        m, lo, hi = paired_bootstrap(diffs)
        boot[f"{b}_minus_{a}"] = {"mean_diff": m, "ci95": [lo, hi], "n": len(common)}
        print(f"  {b} - {a}: {m:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  (n={len(common)})")

    with open(os.path.join(outdir, f"metrics_summary_{args.set}.json"), "w") as f:
        json.dump({"config": data["config"], "summary": summary, "bootstrap": boot,
                   "note": "nDCG ideal ranking is pool-relative; no recall computed "
                           "(no exhaustive relevance labels)."}, f, indent=2)
    with open(os.path.join(outdir, f"metrics_per_query_{args.set}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "query_id", "class", "p_strict", "p_lenient", "hit", "mrr",
                    "ndcg", "junk", "garbled"])
        for arm, qm in per_query.items():
            for q, v in qm.items():
                w.writerow([arm, q, qinfo[q], v["p_strict"], v["p_lenient"], v["hit"], v["mrr"],
                            v["ndcg"], v["junk"], v["garbled"]])
    print(f"\nSaved metrics to {outdir}/")


def cmd_agreement(args):
    def read(p):
        with open(p, newline="") as f:
            return {(r["query_id"], r["chunk_id"]): int(r["grade"])
                    for r in csv.DictReader(f) if (r.get("grade") or "").strip() != ""}
    a, b = read(args.a), read(args.b)
    common = sorted(set(a) & set(b))
    if not common:
        sys.exit("No overlapping graded pairs.")
    exact = sum(a[k] == b[k] for k in common) / len(common)
    cats = [0, 1, 2]
    pa = {c: sum(a[k] == c for k in common) / len(common) for c in cats}
    pb = {c: sum(b[k] == c for k in common) / len(common) for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    kappa = (exact - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"Overlap: {len(common)} pairs | exact agreement: {exact:.2f} | Cohen's kappa: {kappa:.2f}")


# ------------------------------------------------------------------- main ---
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--db-dir", default="vectordb")
        sp.add_argument("--collection", required=True)
        sp.add_argument("--force", action="store_true", help="bypass the over-flagging safety stop")

    a = sub.add_parser("audit"); common(a); a.set_defaults(func=cmd_audit)
    r = sub.add_parser("run"); common(r)
    r.add_argument("--arms", default="dense,dense_clean,hybrid_clean")
    r.add_argument("--run-id", required=True)
    r.add_argument("--k", type=int, default=3)
    r.add_argument("--pool-depth", type=int, default=3)
    r.add_argument("--model", default="all-MiniLM-L6-v2")
    r.set_defaults(func=cmd_run)
    m = sub.add_parser("merge-judgments"); m.add_argument("--todo", required=True)
    m.set_defaults(func=cmd_merge)
    s = sub.add_parser("score"); s.add_argument("--run-id", required=True)
    s.add_argument("--set", default="heldout", choices=["heldout", "dev"])
    s.add_argument("--allow-missing", action="store_true"); s.set_defaults(func=cmd_score)
    g = sub.add_parser("agreement"); g.add_argument("--a", required=True)
    g.add_argument("--b", required=True); g.set_defaults(func=cmd_agreement)
    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
