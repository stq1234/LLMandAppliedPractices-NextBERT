#!/usr/bin/env python3
"""Validate and summarize the completed prepared-data downstream evaluation."""
import argparse, json, math
from pathlib import Path
from statistics import fmean
LANGS, SPLITS = ("en", "zh"), ("dev", "test")
TASK_VARIANTS = {"classification": ("V0", "V3"), "qa": ("V0", "V1", "V2", "V3"), "retrieval": ("V0", "V1", "V2", "V3"), "reranking": ("V0", "V3")}
def read(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {exc}") from exc
def pct(value):
    value = float(value)
    if not math.isfinite(value) or value < 0 or value > 100:
        raise ValueError(f"invalid rate/percentage: {value}")
    return round(value * 100 if value <= 1 else value, 4)
def rate(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"expected a 0..1 retrieval/reranking metric, got {value}")
    return round(value, 6)
def artifact(manifest, key):
    item = manifest.get("artifacts", {}).get(key)
    if not item or not item.get("path") or not item.get("sha256"):
        raise ValueError(f"data manifest lacks source artifact {key}")
    return item
def expected_counts(item):
    counts = item.get("question_counts", item.get("language_counts"))
    if not isinstance(counts, dict) or any(lang not in counts for lang in LANGS):
        raise ValueError("manifest lacks per-language source counts")
    return {lang: int(counts[lang]) for lang in LANGS}
def check_provenance(job, root, source_items, variant):
    files = job.get("provenance", {}).get("files", [])
    by_path = {str(Path(row.get("path", "")).resolve()): row for row in files}
    sources = []
    for key, item in source_items:
        target = str((root / item["path"]).resolve())
        row = by_path.get(target)
        if not row or row.get("sha256") != item["sha256"]:
            raise ValueError(f"{job.get('job_id')}: source provenance mismatch: {key}")
        sources.append({"key": key, "path": item["path"], "sha256": item["sha256"],
                        "rows": item.get("rows")})
    checkpoint_rows = [row for row in files if str(row.get("path", "")).endswith("_checkpoint.pt")]
    if len(checkpoint_rows) != 1:
        raise ValueError(f"{job.get('job_id')}: expected one checkpoint in provenance")
    cp = checkpoint_rows[0]
    cp_path = str(cp["path"]).replace("\\", "/")
    is_gatefix = "/formal_67m_gatefix/" in cp_path
    if is_gatefix != (variant in {"V2", "V3"}):
        raise ValueError(f"{job.get('job_id')}: wrong checkpoint family: {cp_path}")
    return sources, {"path": cp["path"], "sha256": cp["sha256"],
                     "family": "gatefix_downstream" if is_gatefix else "original_downstream"}
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--evaluation-dir", type=Path)
    args = parser.parse_args()
    root = args.root.resolve(); outdir = (args.evaluation_dir or root / "results" / "formal_67m" / "evaluation").resolve()
    try:
        status = read(outdir / "evaluation_status.json")
        manifest = read(root / "data" / "prepared" / "full_67m" / "manifest.json")
        if (status.get("runner"), status.get("phase"), status.get("status")) != ("run_all_evaluation", "tasks", "completed"):
            raise ValueError("evaluation status is incomplete or not the tasks phase")
        expected_jobs = {f"tasks_{v}_{t}_{s}": (t, v, s) for t, variants in TASK_VARIANTS.items()
                         for v in variants for s in (("dev_test",) if t == "retrieval" else SPLITS)}
        jobs = status.get("jobs", {})
        if set(jobs) != set(expected_jobs):
            raise ValueError(f"wrong/incomplete job matrix; missing={sorted(set(expected_jobs)-set(jobs))}, extra={sorted(set(jobs)-set(expected_jobs))}")
        for jid, job in jobs.items():
            if job.get("status") != "succeeded" or job.get("return_code") != 0:
                raise ValueError(f"job did not succeed: {jid}")
        summary = {task: {} for task in TASK_VARIANTS}
        for jid, (task, variant, job_split) in expected_jobs.items():
            job = jobs[jid]
            payload_path = Path(job["output"]); payload = read(payload_path if payload_path.is_absolute() else root / payload_path)
            if payload.get("task") != task or payload.get("variant") != variant or payload.get("evaluation_scope") != "prepared_data_scope":
                raise ValueError(f"wrong result identity/scope: {jid}")
            if task == "retrieval":
                split_payloads = {split: payload for split in SPLITS}
            else:
                split_payloads = {job_split: payload}
            for split, result in split_payloads.items():
                key = {"classification": f"classification_{split}", "qa": f"qa_{split}",
                       "retrieval": f"retrieval_queries_{split}", "reranking": f"reranking_{split}"}[task]
                source_items = [(key, artifact(manifest, key))] + ([
                    ("retrieval_corpus", artifact(manifest, "retrieval_corpus"))]
                    if task in {"retrieval", "reranking"} else [])
                sources, checkpoint = check_provenance(job, root, source_items, variant)
                counts = expected_counts(source_items[0][1])
                if task == "classification":
                    if int(result.get("count", -1)) != sum(counts.values()): raise ValueError(f"{jid}: total count mismatch")
                    langs = result["by_language"]
                    vals = {lang: {"examples": int(langs[lang]["count"]),
                                   "accuracy_pct": pct(langs[lang]["accuracy"]),
                                   "macro_f1_pct": pct(langs[lang]["macro_f1"])} for lang in LANGS}
                    for lang in LANGS:
                        if vals[lang]["examples"] != counts[lang]: raise ValueError(f"{jid}: {lang} count mismatch")
                    metrics = {"balanced_language_macro": {m: round(fmean(vals[l][m] for l in LANGS), 4)
                               for m in ("accuracy_pct", "macro_f1_pct")}, "by_language": vals}
                elif task == "qa":
                    if int(result.get("question_count", -1)) != sum(counts.values()): raise ValueError(f"{jid}: question count mismatch")
                    predictions = result["predictions"]
                    if len(predictions) != sum(counts.values()): raise ValueError(f"{jid}: predictions missing")
                    if not math.isclose(fmean(float(p["f1"]) for p in predictions), float(result["f1"]), abs_tol=1e-12):
                        raise ValueError(f"{jid}: overall F1 inconsistent with saved predictions")
                    vals = {}
                    for lang in LANGS:
                        em, f1 = result["exact_match_by_language"][lang], result["f1_by_language"][lang]
                        if int(em["count"]) != counts[lang] or int(f1["count"]) != counts[lang]: raise ValueError(f"{jid}: {lang} question count mismatch")
                        group = [p for p in predictions if p["language"] == lang]
                        if len(group) != counts[lang]: raise ValueError(f"{jid}: {lang} predictions missing")
                        vals[lang] = {"questions": counts[lang], "EM_pct": pct(fmean(float(p["exact_match"]) for p in group)),
                                      "F1_pct": pct(fmean(float(p["f1"]) for p in group))}
                    metrics = {"questions": sum(counts.values()), "overflow_windows": int(result["overflow_windows"]),
                               "scoring_status": result.get("scoring_status"), "balanced_language_macro":
                               {m: round(fmean(vals[l][m] for l in LANGS), 4) for m in ("EM_pct", "F1_pct")},
                               "by_language": vals}
                elif task == "retrieval":
                    bylang = result["by_language"]
                    total = int(result["queries_by_split"].get(split, -1))
                    if total != sum(counts.values()): raise ValueError(f"{jid}/{split}: query count mismatch")
                    vals = {}
                    for lang in LANGS:
                        item = bylang[lang][split]
                        if int(item["queries"]) != counts[lang]: raise ValueError(f"{jid}/{split}: {lang} query count mismatch")
                        vals[lang] = {"queries": counts[lang], **{m: rate(item[m]) for m in ("nDCG@10", "Recall@10", "Recall@100")}}
                    pools = {str(row["language"]): row for row in result.get("languages", [])}
                    if any(int(pools[l]["documents_indexed"]) != 5000 or pools[l]["sampled_index_only"] for l in LANGS):
                        raise ValueError(f"{jid}: retrieval corpus is not the full prepared 5000-per-language pool")
                    metrics = {"queries": total, "corpus_documents_per_language": 5000,
                               "balanced_language_macro": {m: round(fmean(vals[l][m] for l in LANGS), 6)
                               for m in ("nDCG@10", "Recall@10", "Recall@100")}, "by_language": vals}
                else:
                    if int(result.get("query_count", -1)) != sum(counts.values()): raise ValueError(f"{jid}: query count mismatch")
                    if float(result.get("candidate_count_mean", -1)) != 20.0: raise ValueError(f"{jid}: candidate count is not 20")
                    vals = {}
                    for lang in LANGS:
                        item = result["by_language"][lang]
                        if int(item["queries"]) != counts[lang]: raise ValueError(f"{jid}: {lang} query count mismatch")
                        vals[lang] = {"queries": counts[lang], **{m: rate(item[m]) for m in ("nDCG@10", "MRR@10", "candidate_Recall@20")}}
                    metrics = {"queries": sum(counts.values()), "candidate_count": 20,
                               "balanced_language_macro": {m: round(fmean(vals[l][m] for l in LANGS), 6)
                               for m in ("nDCG@10", "MRR@10", "candidate_Recall@20")}, "by_language": vals}
                summary[task].setdefault(variant, {})[split] = {**metrics, "sources": sources, "checkpoint": checkpoint}
        print(json.dumps({"status": "complete", "evaluation_dir": str(outdir),
                          "units": {"classification": "percent", "qa": "percent; 0..1 scorer outputs multiplied by 100",
                                    "retrieval": "0..1 ratio", "reranking": "0..1 ratio"},
                          "results": summary,
                          "scope_note": "Retrieval uses the fixed reduced 5,000-document-per-language pool; not the full MLDR corpus."},
                         ensure_ascii=False, separators=(",", ":")))
        return 0
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "incomplete_or_invalid", "error": str(exc)}, ensure_ascii=False, separators=(",", ":")))
        return 2
if __name__ == "__main__":
    raise SystemExit(main())
