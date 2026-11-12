"""Score the original BM25 candidate order without model inference."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def evaluate(path):
    groups = defaultdict(list)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            labels, scores = row["labels"], row["bm25_scores"]
            if len(labels) != 20 or any(a < b for a, b in zip(scores, scores[1:])):
                raise ValueError("expected 20 candidates in descending BM25 score order")
            if row["candidate_source"] != "fixed BM25 top-20 over retrieval_corpus.jsonl":
                raise ValueError("candidate order is not the fixed BM25 baseline")
            def dcg(values):
                return sum((2 ** int(v) - 1) / math.log2(i + 2) for i, v in enumerate(values[:10]))
            ideal = dcg(sorted(labels, reverse=True))
            qrels = set(map(str, row["relevant_doc_ids"]))
            present = qrels & set(map(str, row["candidate_docids"]))
            groups[row["language"]].append({
                "nDCG@10": dcg(labels) / ideal if ideal else 0.0,
                "MRR@10": next((1 / (i + 1) for i, v in enumerate(labels[:10]) if v), 0.0),
                "candidate_Recall@20": len(present) / len(qrels) if qrels else 0.0,
            })
    by_language = {lang: {"queries": len(rows), **{
        key: statistics.mean(row[key] for row in rows) for key in rows[0]
    }} for lang, rows in sorted(groups.items())}
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"method": "original_BM25_candidate_order", "source": str(path), "source_sha256": digest,
            "by_language": by_language, "balanced_language_macro": {
                key: statistics.mean(row[key] for row in by_language.values())
                for key in ("nDCG@10", "MRR@10", "candidate_Recall@20")},
            "scope": "same 20 candidates used by the learned rerankers; no model inference"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] /
                        "data/prepared/full_67m/reranking_test.jsonl")
    print(json.dumps(evaluate(parser.parse_args().data), ensure_ascii=False, indent=2))
