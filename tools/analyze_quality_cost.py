#!/usr/bin/env python3
"""Paired CPU analysis of saved NextBERT evaluation outputs.

The script never runs a model. It streams result JSON from the frozen tar archive,
recomputes paired estimates, and optionally consumes later encoder-cost and long
context JSON outputs. Outputs are machine-readable JSON/CSV plus two plots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

LANGUAGES = ("en", "zh")
TASK_VARIANTS = {
    "classification": ("V0", "V3"),
    "qa": ("V0", "V1", "V2", "V3"),
    "retrieval": ("V0", "V1", "V2", "V3"),
    "reranking": ("V0", "V3"),
}
COMPARISONS = {
    "classification": (("V0", "V3"),),
    "qa": (("V0", "V1"), ("V1", "V2"), ("V2", "V3"), ("V0", "V3")),
    "retrieval": (("V0", "V1"), ("V1", "V2"), ("V2", "V3"), ("V0", "V3")),
    "reranking": (("V0", "V3"),),
}
CLASS_METRICS = ("accuracy", "macro_f1")
QA_METRICS = ("EM", "f1")
RETRIEVAL_METRICS = ("nDCG@10", "Recall@10", "Recall@100")
RERANK_METRICS = ("nDCG@10", "MRR@10", "candidate_Recall@20")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSON {path}: {exc}") from exc


def _tar_json(archive: tarfile.TarFile, name: str) -> dict[str, Any]:
    member = archive.extractfile(name)
    if member is None:
        raise ValueError(f"archive member missing: {name}")
    try:
        value = json.load(member)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid result JSON in archive member {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"archive result is not an object: {name}")
    return value


def load_result_archive(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Read current task result members, excluding preserved pre-correction QA copies."""
    found: dict[tuple[str, str, str], dict[str, Any]] = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            name = member.name
            if not member.isfile() or "/tasks/" not in name or not name.endswith(".json"):
                continue
            if "before_language_f1_fix" in name:
                continue
            parts = name.split("/")
            task_index = parts.index("tasks")
            if len(parts) <= task_index + 2:
                continue
            variant = parts[task_index + 1]
            stem = Path(parts[-1]).stem
            pieces = stem.split("_")
            task = pieces[0]
            split = "_".join(pieces[1:])
            if task not in TASK_VARIANTS or variant not in TASK_VARIANTS[task]:
                continue
            if task == "retrieval":
                if split != "dev_test":
                    continue
                split = "combined"
            elif split not in {"dev", "test"}:
                continue
            key = (task, variant, split)
            if key in found:
                raise ValueError(f"duplicate task output for {key}: {name}")
            result = _tar_json(archive, name)
            if (result.get("task"), result.get("variant")) != (task, variant):
                raise ValueError(f"wrong task identity in {name}")
            if result.get("evaluation_scope") != "prepared_data_scope":
                raise ValueError(f"unexpected evaluation scope in {name}: {result.get('evaluation_scope')}")
            result["_archive_member"] = name
            found[key] = result
    expected = {
        (task, variant, split)
        for task, variants in TASK_VARIANTS.items()
        for variant in variants
        for split in (("combined",) if task == "retrieval" else ("dev", "test"))
    }
    if set(found) != expected:
        raise ValueError(f"incomplete task archive; missing={sorted(expected-set(found))}, extra={sorted(set(found)-expected)}")
    return found


def macro_f1(labels: Sequence[int], predictions: Sequence[int], classes: Sequence[int] = (0, 1, 2)) -> float:
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("macro_f1 requires same-sized nonempty label and prediction arrays")
    values = []
    for label in classes:
        tp = int(np.count_nonzero((y == label) & (p == label)))
        denom = int(np.count_nonzero(y == label)) + int(np.count_nonzero(p == label))
        values.append((2.0 * tp / denom) if denom else 0.0)
    return float(np.mean(values))


def _bootstrap_units(
    ids: Sequence[str], languages: Mapping[str, str],
    clusters: Mapping[str, str] | None, rng: np.random.Generator, reps: int,
    sync_languages: bool = False,
) -> dict[str, list[np.ndarray]]:
    result: dict[str, list[np.ndarray]] = {}
    ids_arr = np.asarray(ids, dtype=object)
    langs_arr = np.asarray([languages[x] for x in ids], dtype=object)
    cluster_ids = np.asarray([clusters[x] if clusters else x for x in ids], dtype=object)
    unit_positions_by_language: dict[str, dict[str, list[int]]] = {}
    for language in LANGUAGES:
        lang_positions = np.flatnonzero(langs_arr == language)
        if not len(lang_positions):
            raise ValueError(f"missing language {language}")
        unit_to_positions: dict[str, list[int]] = defaultdict(list)
        for pos in lang_positions:
            unit_to_positions[str(cluster_ids[pos])].append(int(pos))
        unit_positions_by_language[language] = unit_to_positions
    if sync_languages:
        common_units = set.intersection(*(set(unit_positions_by_language[lang]) for lang in LANGUAGES))
        units = np.asarray(sorted(common_units), dtype=object)
        if not len(units):
            raise ValueError("synchronized language bootstrap has no shared clusters")
        per_language_samples: dict[str, list[np.ndarray]] = {lang: [] for lang in LANGUAGES}
        for _ in range(reps):
            selected = rng.choice(units, size=len(units), replace=True)
            for language in LANGUAGES:
                positions = unit_positions_by_language[language]
                per_language_samples[language].append(np.asarray(
                    [p for unit in selected for p in positions[str(unit)]], dtype=np.int64))
        return per_language_samples
    for language in LANGUAGES:
        unit_to_positions = unit_positions_by_language[language]
        units = np.asarray(sorted(unit_to_positions), dtype=object)
        if not len(units):
            raise ValueError(f"no resampling units in {language}")
        samples = []
        for _ in range(reps):
            selected = rng.choice(units, size=len(units), replace=True)
            samples.append(np.asarray([p for unit in selected for p in unit_to_positions[str(unit)]], dtype=np.int64))
        result[language] = samples
    return result


def paired_bootstrap(
    task: str,
    rows_by_variant: Mapping[str, Mapping[str, Mapping[str, Any]]],
    metric_names: Sequence[str],
    metric_fn: Callable[[Sequence[Mapping[str, Any]], str], float],
    *,
    reps: int,
    seed: int,
    clusters: Mapping[str, str] | None = None,
    point_unit: str,
    scale: float = 1.0,
    comparisons: Sequence[tuple[str, str]] | None = None,
    sync_languages: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    variants = list(rows_by_variant)
    if not variants:
        raise ValueError(f"no variants for {task}")
    id_sets = [set(rows_by_variant[v]) for v in variants]
    if any(s != id_sets[0] for s in id_sets[1:]):
        missing = {v: sorted(id_sets[0] - set(rows_by_variant[v]))[:5] for v in variants[1:]}
        extra = {v: sorted(set(rows_by_variant[v]) - id_sets[0])[:5] for v in variants[1:]}
        raise ValueError(f"unpaired IDs for {task}; missing={missing}; extra={extra}")
    ids = sorted(id_sets[0])
    first = rows_by_variant[variants[0]]
    languages = {qid: str(first[qid]["language"]) for qid in ids}
    for variant in variants[1:]:
        for qid in ids:
            if str(rows_by_variant[variant][qid]["language"]) != languages[qid]:
                raise ValueError(f"language mismatch for paired id {qid}")
    if clusters is not None and any(qid not in clusters for qid in ids):
        raise ValueError(f"missing cluster mapping for {task} IDs")
    draws = _bootstrap_units(ids, languages, clusters, np.random.default_rng(seed), reps, sync_languages)
    points: list[dict[str, Any]] = []
    boot: dict[str, dict[str, dict[str, np.ndarray]]] = {v: {} for v in variants}
    grouped_ids = {lang: [qid for qid in ids if languages[qid] == lang] for lang in LANGUAGES}
    for variant in variants:
        row_map = rows_by_variant[variant]
        for language in LANGUAGES:
            sample_ids = grouped_ids[language]
            rows = [row_map[qid] for qid in sample_ids]
            boot[variant][language] = {}
            for metric in metric_names:
                estimate = metric_fn(rows, metric) * scale
                replicates = np.asarray([
                    metric_fn([row_map[ids[i]] for i in index], metric) * scale
                    for index in draws[language]
                ], dtype=np.float64)
                boot[variant][language][metric] = replicates
                low, high = np.quantile(replicates, [0.025, 0.975])
                points.append({"task": task, "variant": variant, "language": language,
                               "metric": metric, "estimate": estimate, "ci_low": float(low),
                               "ci_high": float(high), "unit": "percent" if scale == 100 else "ratio",
                               "observations": len(rows), "resampling_units": len(set(clusters[q] for q in sample_ids)) if clusters else len(rows),
                               "bootstrap_replicates": reps, "resampling_unit": point_unit,
                               "ci_status": "paired_bootstrap_95pct"})
        for metric in metric_names:
            balanced = (boot[variant]["en"][metric] + boot[variant]["zh"][metric]) / 2.0
            all_rows = [row_map[qid] for qid in ids]
            estimate = (metric_fn([row_map[q] for q in grouped_ids["en"]], metric) +
                        metric_fn([row_map[q] for q in grouped_ids["zh"]], metric)) * scale / 2.0
            low, high = np.quantile(balanced, [0.025, 0.975])
            points.append({"task": task, "variant": variant, "language": "en_zh_macro",
                           "metric": metric, "estimate": estimate, "ci_low": float(low),
                           "ci_high": float(high), "unit": "percent" if scale == 100 else "ratio",
                           "observations": len(all_rows), "resampling_units": sum(
                               len(set(clusters[q] for q in grouped_ids[lang])) if clusters else len(grouped_ids[lang])
                               for lang in LANGUAGES), "bootstrap_replicates": reps,
                           "resampling_unit": point_unit, "ci_status": "paired_bootstrap_95pct"})
    deltas: list[dict[str, Any]] = []
    for left, right in (comparisons or ()):
        if left not in boot or right not in boot:
            raise ValueError(f"comparison {left}->{right} unavailable for {task}")
        for metric in metric_names:
            per_lang = {}
            for language in LANGUAGES:
                a = [rows_by_variant[left][q] for q in grouped_ids[language]]
                b = [rows_by_variant[right][q] for q in grouped_ids[language]]
                point_delta = (metric_fn(b, metric) - metric_fn(a, metric)) * scale
                delta_draws = boot[right][language][metric] - boot[left][language][metric]
                lo, hi = np.quantile(delta_draws, [0.025, 0.975])
                per_lang[language] = delta_draws
                deltas.append({"task": task, "comparison": f"{right}-{left}", "left": left, "right": right,
                               "language": language, "metric": metric, "delta": point_delta,
                               "ci_low": float(lo), "ci_high": float(hi),
                               "unit": "percentage_points" if scale == 100 else "ratio",
                               "observations": len(a), "resampling_unit": point_unit,
                               "bootstrap_replicates": reps, "ci_status": "paired_bootstrap_95pct"})
            macro_draws = (per_lang["en"] + per_lang["zh"]) / 2.0
            left_en = [rows_by_variant[left][q] for q in grouped_ids["en"]]
            right_en = [rows_by_variant[right][q] for q in grouped_ids["en"]]
            left_zh = [rows_by_variant[left][q] for q in grouped_ids["zh"]]
            right_zh = [rows_by_variant[right][q] for q in grouped_ids["zh"]]
            point_delta = ((metric_fn(right_en, metric) - metric_fn(left_en, metric)) +
                           (metric_fn(right_zh, metric) - metric_fn(left_zh, metric))) * scale / 2.0
            lo, hi = np.quantile(macro_draws, [0.025, 0.975])
            deltas.append({"task": task, "comparison": f"{right}-{left}", "left": left, "right": right,
                           "language": "en_zh_macro", "metric": metric, "delta": point_delta,
                           "ci_low": float(lo), "ci_high": float(hi),
                           "unit": "percentage_points" if scale == 100 else "ratio",
                           "observations": len(ids), "resampling_unit": point_unit,
                           "bootstrap_replicates": reps, "ci_status": "paired_bootstrap_95pct"})
    return points, deltas


def _classification_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    labels = [int(row["label"]) for row in rows]
    predictions = [int(row["prediction"]) for row in rows]
    if metric == "accuracy":
        return float(np.mean(np.asarray(labels) == np.asarray(predictions)))
    if metric == "macro_f1":
        return macro_f1(labels, predictions)
    raise ValueError(f"unknown classification metric {metric}")


def _qa_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    key = "exact_match" if metric == "EM" else "f1"
    return float(np.mean([float(row[key]) for row in rows]))


def _ranking_metric(rows: Sequence[Mapping[str, Any]], metric: str) -> float:
    return float(np.mean([float(row[metric]) for row in rows]))


def _load_question_clusters(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    question_cluster: dict[str, str] = {}
    question_language: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            qid = str(row.get("question_id") or row.get("id") or "")
            language = str(row.get("language") or "")
            if not qid or language not in LANGUAGES:
                raise ValueError(f"bad QA source row {path}:{line_number}")
            source = str(row.get("source_id") or row.get("dataset") or "source")
            title = " ".join(str(row.get("title") or "").split())
            cluster = f"{language}:{source}:{title}" if title else f"{language}:{source}:question:{qid}"
            if qid in question_cluster and (question_cluster[qid] != cluster or question_language[qid] != language):
                raise ValueError(f"QA question ID maps to multiple articles: {qid}")
            question_cluster[qid] = cluster
            question_language[qid] = language
    return question_cluster, question_language


def _load_classification_clusters(path: Path) -> tuple[dict[str, str], dict[str, Any]]:
    rows: dict[str, dict[str, dict[str, Any]]] = {lang: {} for lang in LANGUAGES}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            language = str(row.get("language") or "")
            qid = str(row.get("id") or "")
            index = qid.rsplit(":", 1)[-1]
            if language not in LANGUAGES or not qid or not index.isdigit() or index in rows[language]:
                raise ValueError(f"bad/duplicate XNLI source ID at {path}:{line_number}")
            rows[language][index] = row
    en_ids, zh_ids = set(rows["en"]), set(rows["zh"])
    if en_ids != zh_ids:
        raise ValueError(f"XNLI source-index mismatch: en_only={len(en_ids-zh_ids)}, zh_only={len(zh_ids-en_ids)}")
    mismatch = [index for index in en_ids if int(rows["en"][index]["label"]) != int(rows["zh"][index]["label"])]
    if mismatch:
        raise ValueError(f"XNLI translated labels differ for {len(mismatch)} aligned indices")
    en_premise = {index: " ".join(str(row.get("premise") or "").split()) for index, row in rows["en"].items()}
    premise_hash = {index: hashlib.sha256((premise or f"source-index:{index}").encode("utf-8")).hexdigest()
                    for index, premise in en_premise.items()}
    clusters = {}
    for language in LANGUAGES:
        for index, row in rows[language].items():
            clusters[str(row["id"])] = premise_hash[index]
    premise_counts: dict[str, int] = defaultdict(int)
    for group in premise_hash.values():
        premise_counts[group] += 1
    return clusters, {
        "rows_per_language": {lang: len(rows[lang]) for lang in LANGUAGES},
        "aligned_terminal_indices": len(en_ids), "label_mismatches": 0,
        "english_premise_clusters": len(premise_counts),
        "premise_groups_repeated_across_source_indices": sum(n > 1 for n in premise_counts.values()),
        "maximum_source_indices_per_premise": max(premise_counts.values()),
        "resampling_unit": "english_premise_cluster_shared_across_aligned_en_zh_ids",
    }


def _result_predictions(result: Mapping[str, Any], *, task: str) -> dict[str, dict[str, Any]]:
    predictions = result.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError(f"{task} result lacks per-example predictions")
    output = {}
    for row in predictions:
        qid = str(row.get("id") or "")
        if not qid or qid in output:
            raise ValueError(f"{task} result has missing/duplicate prediction ID: {qid}")
        language = str(row.get("language") or "")
        if language not in LANGUAGES:
            raise ValueError(f"{task} result has invalid language for {qid}: {language}")
        output[qid] = dict(row, id=qid, language=language)
    return output


def _csv_crosscheck(csv_path: Path, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    index = {(r["variant"], r["task"], r["split"], r["language"], r["metric"]): r for r in csv_rows}
    checked = 0
    for row in rows:
        task, metric = row["task"], row["metric"]
        csv_metric = {"accuracy": "accuracy_pct", "macro_f1": "macro_f1_pct", "EM": "EM_pct", "f1": "F1_pct"}.get(metric, metric)
        key = (row["variant"], task, row["split"], row["language"], csv_metric)
        reference = index.get(key)
        if reference is None:
            continue
        value = float(reference["value"])
        if not math.isclose(value, float(row["estimate"]), rel_tol=0.0, abs_tol=0.001):
            raise ValueError(f"metrics.csv disagrees with prediction-derived point: {key}: {value} != {row['estimate']}")
        checked += 1
    if checked == 0:
        raise ValueError(f"no point estimates cross-checked against {csv_path}")
    return {"status": "matched", "rows_checked": checked, "file": str(csv_path)}


def _bm25_per_query(path: Path) -> dict[str, dict[str, Any]]:
    """Reproduce the saved BM25 candidate-order metrics without model inference."""
    output = {}
    def dcg(labels: Sequence[int]) -> float:
        return sum((2 ** int(value) - 1) / math.log2(i + 2) for i, value in enumerate(labels[:10]))
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            qid = str(row.get("id") or "")
            labels, scores = row.get("labels"), row.get("bm25_scores")
            if not qid or qid in output or len(labels or []) != 20 or len(scores or []) != 20:
                raise ValueError(f"invalid BM25 candidate row {path}:{line_number}")
            if any(a < b for a, b in zip(scores, scores[1:])):
                raise ValueError(f"BM25 scores are not sorted descending at {qid}")
            if row.get("candidate_source") != "fixed BM25 top-20 over retrieval_corpus.jsonl":
                raise ValueError(f"unexpected BM25 candidate source at {qid}")
            qrels = set(map(str, row.get("relevant_doc_ids") or []))
            present = qrels.intersection(map(str, row.get("candidate_docids") or []))
            ideal = dcg(sorted(labels, reverse=True))
            output[qid] = {
                "id": qid, "language": str(row["language"]),
                "nDCG@10": dcg(labels) / ideal if ideal else 0.0,
                "MRR@10": next((1.0 / (i + 1) for i, value in enumerate(labels[:10]) if value), 0.0),
                "candidate_Recall@20": len(present) / len(qrels) if qrels else 0.0,
            }
    return output


def _summarized_point(result: Mapping[str, Any], task: str, variant: str, split: str,
                      language: str, metric: str) -> float:
    if task == "classification":
        key = {"accuracy": "accuracy_pct", "macro_f1": "macro_f1_pct"}[metric]
        return float(result["by_language"][language][key]) / 100.0
    if task == "qa":
        key = "EM_pct" if metric == "EM" else "F1_pct"
        return float(result["by_language"][language][key]) / 100.0
    return float(result["by_language"][language][metric])


def _validate_summary(summary: Mapping[str, Any], archive_results: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> int:
    checked = 0
    metrics = {
        "classification": ("accuracy", "macro_f1"), "qa": QA_METRICS,
        "retrieval": RETRIEVAL_METRICS, "reranking": RERANK_METRICS,
    }
    for task, variants in TASK_VARIANTS.items():
        for variant in variants:
            splits = ("dev", "test") if task != "retrieval" else ("dev", "test")
            for split in splits:
                key = (task, variant, "combined" if task == "retrieval" else split)
                result = archive_results[key]
                for language in LANGUAGES:
                    if task in {"classification", "qa", "reranking"}:
                        rows = _result_predictions(result, task=task)
                        lang_rows = [r for r in rows.values() if r["language"] == language]
                        for metric in metrics[task]:
                            estimate = _ranking_metric(lang_rows, metric) if task == "reranking" else None
                            if task == "classification":
                                estimate = _classification_metric(lang_rows, metric)
                            elif task == "qa":
                                estimate = _qa_metric(lang_rows, metric)
                            target = _summarized_point(summary["results"][task][variant][split], task, variant, split, language, metric)
                            if not math.isclose(float(estimate), target, rel_tol=0.0, abs_tol=0.00012):
                                raise ValueError(f"summary vs raw prediction mismatch: {task}/{variant}/{split}/{language}/{metric}: {estimate} != {target}")
                            checked += 1
                    elif task == "retrieval":
                        for metric in metrics[task]:
                            target = _summarized_point(summary["results"][task][variant][split], task, variant, split, language, metric)
                            rows = result.get("by_language", {}).get(language, {}).get(split)
                            if not rows or not math.isclose(float(rows[metric]), target, rel_tol=0.0, abs_tol=1e-6):
                                raise ValueError(f"summary vs retrieval result mismatch: {variant}/{split}/{language}/{metric}")
                            checked += 1
    return checked


def _metric_record(task: str, split: str, variant: str, language: str, metric: str, estimate: float,
                   count: int, unit: str, ci_low: float | None = None, ci_high: float | None = None,
                   status: str = "point_only", resampling_unit: str | None = None,
                   resampling_units: int | None = None) -> dict[str, Any]:
    return {"task": task, "split": split, "variant": variant, "language": language, "metric": metric,
            "estimate": float(estimate), "ci_low": ci_low, "ci_high": ci_high, "unit": unit,
            "observations": count, "resampling_units": resampling_units,
            "resampling_unit": resampling_unit, "ci_status": status}


def analyze_saved_tasks(
    archive_results: Mapping[tuple[str, str, str], dict[str, Any]],
    summary: Mapping[str, Any],
    qa_data_dir: Path,
    reranking_data_dir: Path,
    bm25_baseline_path: Path,
    *, reps: int, seed: int, splits: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    metric_rows: list[dict[str, Any]] = []
    delta_rows: list[dict[str, Any]] = []
    notes = ["Bilingual aggregates weight en and zh equally; reported task point estimates are recomputed from the saved per-example outputs."]
    for split in splits:
        for task, variants in TASK_VARIANTS.items():
            if task == "retrieval":
                by_variant = {}
                for variant in variants:
                    result = summary["results"][task][variant][split]
                    counts = {}
                    for language in LANGUAGES:
                        item = result["by_language"][language]
                        counts[language] = int(item["queries"])
                        for metric in RETRIEVAL_METRICS:
                            metric_rows.append(_metric_record(task, split, variant, language, metric,
                                float(item[metric]), counts[language], "ratio", status="unavailable_saved_query_metrics"))
                    for metric in RETRIEVAL_METRICS:
                        balanced = float(result["balanced_language_macro"][metric])
                        metric_rows.append(_metric_record(task, split, variant, "en_zh_macro", metric,
                            balanced, sum(counts.values()), "ratio", status="unavailable_saved_query_metrics"))
                    by_variant[variant] = result
                for left, right in COMPARISONS[task]:
                    for metric in RETRIEVAL_METRICS:
                        for language in (*LANGUAGES, "en_zh_macro"):
                            a = float(by_variant[left]["by_language"][language][metric]) if language != "en_zh_macro" else float(by_variant[left]["balanced_language_macro"][metric])
                            b = float(by_variant[right]["by_language"][language][metric]) if language != "en_zh_macro" else float(by_variant[right]["balanced_language_macro"][metric])
                            delta_rows.append({"task": task, "split": split, "comparison": f"{right}-{left}",
                                "left": left, "right": right, "language": language, "metric": metric,
                                "delta": b-a, "ci_low": None, "ci_high": None, "unit": "ratio",
                                "observations": None, "resampling_unit": "query", "bootstrap_replicates": reps,
                                "ci_status": "unavailable_saved_query_metrics"})
                notes.append("Retrieval retains saved point estimates and per-language counts; the archive's combined retrieval JSON omits query_predictions, so paired query bootstrap CIs are unavailable.")
                continue

            predictions = {v: _result_predictions(archive_results[(task, v, split)], task=task) for v in variants}
            row_metrics = CLASS_METRICS if task == "classification" else QA_METRICS if task == "qa" else RERANK_METRICS
            cluster_map = None
            resampling_unit = "example"
            sync_languages = False
            if task == "qa":
                cluster_map, source_languages = _load_question_clusters(qa_data_dir / f"qa_{split}.jsonl")
                for qid, row in next(iter(predictions.values())).items():
                    if qid not in cluster_map or source_languages[qid] != row["language"]:
                        raise ValueError(f"QA prediction {qid} does not align with source article metadata")
                resampling_unit = "source_article_cluster"
            elif task == "classification":
                cluster_map, cluster_info = _load_classification_clusters(qa_data_dir / f"classification_{split}.jsonl")
                for qid, row in next(iter(predictions.values())).items():
                    if qid not in cluster_map:
                        raise ValueError(f"classification prediction {qid} does not align with XNLI premise source")
                resampling_unit = "synchronized_en_zh_english_premise_cluster"
                sync_languages = True
                notes.append("XNLI bootstrap synchronously resamples shared English-premise clusters across EN/ZH; "
                             f"source check: {json.dumps(cluster_info, ensure_ascii=False, separators=(',', ':'))}.")
            if task == "classification":
                metric_fn, scale = _classification_metric, 100.0
            elif task == "qa":
                metric_fn, scale = _qa_metric, 100.0
            else:
                metric_fn, scale = _ranking_metric, 1.0
            point_rows, pair_rows = paired_bootstrap(task, predictions, row_metrics, metric_fn,
                reps=reps, seed=seed, clusters=cluster_map, point_unit=resampling_unit,
                scale=scale, comparisons=COMPARISONS[task], sync_languages=sync_languages)
            for row in point_rows:
                row["split"] = split
                metric_rows.append(row)
            for row in pair_rows:
                row["split"] = split
                delta_rows.append(row)

        # Recompute the BM25 candidate-order baseline and pair it with both rerankers.
        bm25_rows = _bm25_per_query(reranking_data_dir / f"reranking_{split}.jsonl")
        for variant in ("V0", "V3"):
            model_rows = _result_predictions(archive_results[("reranking", variant, split)], task="reranking")
            if set(model_rows) != set(bm25_rows):
                raise ValueError(f"BM25/model query IDs differ for {split}/{variant}")
            for qid in bm25_rows:
                if model_rows[qid]["language"] != bm25_rows[qid]["language"]:
                    raise ValueError(f"BM25/model language mismatch for query {qid}")
            if variant == "V0" and split == "test":
                saved = _read_json(bm25_baseline_path)
                for language in LANGUAGES:
                    subset = [r for r in bm25_rows.values() if r["language"] == language]
                    for metric in RERANK_METRICS:
                        value = _ranking_metric(subset, metric)
                        target = float(saved["by_language"][language][metric])
                        if not math.isclose(value, target, rel_tol=0.0, abs_tol=1e-12):
                            raise ValueError(f"recomputed BM25 disagrees with saved baseline for {language}/{metric}: {value} != {target}")
            baseline_map = {"BM25": bm25_rows, variant: model_rows}
            point_rows, pair_rows = paired_bootstrap("reranking_BM25", baseline_map, RERANK_METRICS,
                _ranking_metric, reps=reps, seed=seed, point_unit="query", scale=1.0,
                comparisons=(("BM25", variant),))
            # The comparison direction above is model - BM25.
            for row in point_rows:
                row.update(task="reranking", split=split, source="frozen_BM25_top20")
                if row["variant"] == "BM25" and variant == "V0":
                    metric_rows.append(row)
            for row in pair_rows:
                row.update(task="reranking", split=split, comparison=f"{variant}-BM25", left="BM25", right=variant,
                           source="frozen_BM25_top20")
                delta_rows.append(row)
        notes.append("BM25 per-query scores are recomputed on the saved fixed top-20 candidate order; test point estimates are checked against bm25_baseline.json.")
    return metric_rows, delta_rows, notes


def _long_values(rows: Sequence[Mapping[str, Any]], condition: str, field: str) -> dict[str, dict[str, dict[str, float]]]:
    values: dict[str, dict[str, dict[str, float]]] = {lang: {pos: {} for pos in ("first", "middle", "last")} for lang in LANGUAGES}
    for row in rows:
        if (row.get("condition"), row.get("view"), int(row.get("length", -1))) != (condition, "full", 8192):
            continue
        lang, pos, group = str(row.get("language")), str(row.get("evidence_position")), str(row.get("source_group_id"))
        value = row.get(field)
        if lang not in LANGUAGES or pos not in values[lang] or value is None:
            continue
        if group in values[lang][pos]:
            raise ValueError(f"duplicate long-context source group/cell: {lang}/{group}/{pos}/{condition}")
        values[lang][pos][group] = float(value)
    return values


def _long_breakdown(rows_by_variant: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int, str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for variant, rows in rows_by_variant.items():
        for row in rows:
            key = (variant, str(row.get("language")), int(row.get("length", 0)),
                   str(row.get("evidence_position")), str(row.get("condition")), str(row.get("view")))
            grouped[key].append(row)
    output = []
    for (variant, language, length, position, condition, view), rows in sorted(grouped.items()):
        record = {"variant": variant, "language": language, "length": length,
                  "evidence_position": position, "condition": condition, "view": view,
                  "predictions": len(rows), "source_groups": len({str(r["source_group_id"]) for r in rows})}
        if condition == "evidence_deleted":
            record.update({"exact_match": None, "f1": None,
                           "old_answer_reproduction_rate": statistics.mean(1.0-float(r["fact_change_correct"]) for r in rows),
                           "metric_note": "exact match to the original answer after deleting evidence; not refusal accuracy"})
        else:
            scored = [r for r in rows if r.get("exact_match") is not None and r.get("f1") is not None]
            record.update({"exact_match": statistics.mean(float(r["exact_match"]) for r in scored) if scored else None,
                           "f1": statistics.mean(float(r["f1"]) for r in scored) if scored else None,
                           "old_answer_reproduction_rate": None,
                           "metric_note": "official answer EM/F1 on constructed QA case"})
            record["scored_predictions"] = len(scored)
        output.append(record)
    return output


def _long_condition_analysis(
    values: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]], *,
    task: str, metric: str, transform: Callable[[float], float], reps: int, rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float], dict[str, dict[str, int]], dict[str, float]]:
    positions = ("first", "middle", "last")
    variants = list(values)
    common_pos: dict[str, dict[str, list[str]]] = {lang: {} for lang in LANGUAGES}
    union_groups: dict[str, list[str]] = {}
    common_all: dict[str, list[str]] = {}
    samples_union: dict[str, np.ndarray] = {}
    for lang in LANGUAGES:
        for pos in positions:
            sets = [set(values[v][lang][pos]) for v in variants]
            common = set.intersection(*sets) if sets else set()
            if not common:
                raise ValueError(f"no matched source groups for long context: {lang}/{pos}/{task}")
            common_pos[lang][pos] = sorted(common)
        all_pos_sets = [set(common_pos[lang][pos]) for pos in positions]
        common_all[lang] = sorted(set.intersection(*all_pos_sets))
        union_groups[lang] = sorted(set.union(*all_pos_sets))
        if not union_groups[lang]:
            raise ValueError(f"no matched source groups for long context: {lang}/{task}")
        # Resample each source group once per language, then apply the same draw
        # across all positions and variants. Groups missing a position remain NaN.
        samples = rng.integers(0, len(union_groups[lang]), size=(reps, len(union_groups[lang])))
        position_presence = np.asarray([
            [group in set(common_pos[lang][pos]) for group in union_groups[lang]]
            for pos in positions
        ], dtype=bool)
        for rep in range(reps):
            attempts = 0
            while not position_presence[:, samples[rep]].any(axis=1).all():
                samples[rep] = rng.integers(0, len(union_groups[lang]), size=len(union_groups[lang]))
                attempts += 1
                if attempts >= 1000:
                    raise ValueError(f"could not draw source groups covering all positions for {lang}/{task}")
        samples_union[lang] = samples

    points: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    pos_draws: dict[str, dict[str, dict[str, np.ndarray]]] = {v: {lang: {} for lang in LANGUAGES} for v in variants}
    equal_draws: dict[str, dict[str, np.ndarray]] = {v: {} for v in variants}
    pos_points: dict[str, dict[str, dict[str, float]]] = {v: {lang: {} for lang in LANGUAGES} for v in variants}
    equal_points: dict[str, dict[str, float]] = {v: {} for v in variants}
    complete_case_points: dict[str, dict[str, float]] = {v: {} for v in variants}

    for variant in variants:
        for lang in LANGUAGES:
            union = union_groups[lang]
            sample_idx = samples_union[lang]
            per_position = np.full((len(positions), len(union)), np.nan, dtype=np.float64)
            for pos in positions:
                groups = common_pos[lang][pos]
                raw = np.asarray([transform(values[variant][lang][pos][g]) for g in groups], dtype=np.float64)
                union_idx = {group: index for index, group in enumerate(union)}
                position = positions.index(pos)
                raw_union = np.full(len(union), np.nan, dtype=np.float64)
                raw_union[[union_idx[group] for group in groups]] = raw
                per_position[position] = raw_union
                sampled = raw_union[sample_idx]
                position_draws = np.nanmean(sampled, axis=1)
                draws = position_draws
                pos_draws[variant][lang][pos] = draws
                estimate = float(raw.mean())
                pos_points[variant][lang][pos] = estimate
                lo, hi = np.quantile(draws, [0.025, 0.975])
                points.append(_metric_record(task, "test", variant, lang, f"{metric}_{pos}", estimate,
                    len(groups), "ratio", float(lo), float(hi), "paired_cluster_bootstrap_95pct", "source_group", len(groups)))
            sampled_positions = per_position[:, sample_idx]
            position_means = np.nanmean(sampled_positions, axis=2)
            draws = position_means.mean(axis=0)
            equal_draws[variant][lang] = draws
            estimate = float(np.nanmean(per_position, axis=1).mean())
            equal_points[variant][lang] = estimate
            lo, hi = np.quantile(draws, [0.025, 0.975])
            points.append(_metric_record(task, "test", variant, lang, metric, estimate,
                sum(len(common_pos[lang][pos]) for pos in positions), "ratio", float(lo), float(hi), "paired_cluster_bootstrap_95pct",
                "source_group", len(union)))
            complete = common_all[lang]
            if complete:
                complete_position_means = [
                    statistics.mean(transform(values[variant][lang][pos][group]) for group in complete)
                    for pos in positions
                ]
                complete_case_points[variant][lang] = statistics.mean(complete_position_means)
            else:
                complete_case_points[variant][lang] = math.nan
        macro_draws = (equal_draws[variant]["en"] + equal_draws[variant]["zh"]) / 2.0
        lo, hi = np.quantile(macro_draws, [0.025, 0.975])
        balanced = (equal_points[variant]["en"] + equal_points[variant]["zh"]) / 2.0
        points.append(_metric_record(task, "test", variant, "en_zh_macro", metric, balanced,
            sum(len(common_pos[l][pos]) for l in LANGUAGES for pos in positions), "ratio", float(lo), float(hi),
            "paired_cluster_bootstrap_95pct", "source_group", sum(len(union_groups[l]) for l in LANGUAGES)))

    for left, right in COMPARISONS["qa"]:
        for lang in LANGUAGES:
            for pos in positions:
                draws = pos_draws[right][lang][pos] - pos_draws[left][lang][pos]
                lo, hi = np.quantile(draws, [0.025, 0.975])
                deltas.append({"task": task, "split": "test", "comparison": f"{right}-{left}", "left": left,
                    "right": right, "language": lang, "metric": f"{metric}_{pos}",
                    "delta": pos_points[right][lang][pos] - pos_points[left][lang][pos],
                    "ci_low": float(lo), "ci_high": float(hi), "unit": "ratio",
                    "resampling_unit": "source_group", "bootstrap_replicates": reps,
                    "ci_status": "paired_cluster_bootstrap_95pct"})
            draws = equal_draws[right][lang] - equal_draws[left][lang]
            lo, hi = np.quantile(draws, [0.025, 0.975])
            deltas.append({"task": task, "split": "test", "comparison": f"{right}-{left}", "left": left,
                "right": right, "language": lang, "metric": metric,
                "delta": equal_points[right][lang] - equal_points[left][lang],
                "ci_low": float(lo), "ci_high": float(hi), "unit": "ratio",
                "resampling_unit": "source_group", "bootstrap_replicates": reps,
                "ci_status": "paired_cluster_bootstrap_95pct"})
        macro = ((equal_draws[right]["en"] + equal_draws[right]["zh"]) -
                 (equal_draws[left]["en"] + equal_draws[left]["zh"])) / 2.0
        lo, hi = np.quantile(macro, [0.025, 0.975])
        estimate = ((equal_points[right]["en"] + equal_points[right]["zh"]) -
                    (equal_points[left]["en"] + equal_points[left]["zh"])) / 2.0
        deltas.append({"task": task, "split": "test", "comparison": f"{right}-{left}", "left": left,
            "right": right, "language": "en_zh_macro", "metric": metric, "delta": estimate,
            "ci_low": float(lo), "ci_high": float(hi), "unit": "ratio", "resampling_unit": "source_group",
            "bootstrap_replicates": reps, "ci_status": "paired_cluster_bootstrap_95pct"})
    counts = {lang: {**{pos: len(common_pos[lang][pos]) for pos in positions},
                     "union": len(union_groups[lang]), "complete_case": len(common_all[lang])}
              for lang in LANGUAGES}
    equal_point_by_variant = {v: (equal_points[v]["en"] + equal_points[v]["zh"]) / 2.0 for v in variants}
    complete_case_macro = {
        v: (complete_case_points[v]["en"] + complete_case_points[v]["zh"]) / 2.0
        if math.isfinite(complete_case_points[v]["en"]) and math.isfinite(complete_case_points[v]["zh"])
        else math.nan
        for v in variants
    }
    return points, deltas, equal_point_by_variant, counts, complete_case_macro


def analyze_long_context(
    long_files: Mapping[str, Path], *, reps: int, seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, float], list[dict[str, Any]]]:
    if not long_files:
        return [], [], {"status": "pending_long_json"}, {}, []
    expected_variants = set(TASK_VARIANTS["qa"])
    if set(long_files) != expected_variants:
        raise ValueError(f"long JSONs must include exactly {sorted(expected_variants)}")
    data: dict[str, list[dict[str, Any]]] = {}
    for variant, path in long_files.items():
        result = _read_json(path)
        if result.get("variant") != variant or result.get("diagnostic") != "controlled_long_context_evidence":
            raise ValueError(f"wrong long-context result identity at {path}")
        rows = result.get("predictions")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"long-context JSON lacks predictions: {path}")
        data[variant] = rows
    points: list[dict[str, Any]] = []
    deltas: list[dict[str, Any]] = []
    primary_summary: dict[str, Any] = {}
    primary_point: dict[str, float] = {}
    rng = np.random.default_rng(seed)
    conditions = (
        ("intact", "f1", "F1_equal_position", lambda x: x),
        ("counterfactual", "f1", "F1_equal_position", lambda x: x),
        ("evidence_deleted", "fact_change_correct", "old_answer_reproduction_rate_equal_position", lambda x: 1.0-x),
        ("background_replaced", "f1", "F1_equal_position", lambda x: x),
    )
    condition_info = {}
    for condition, field, metric, transform in conditions:
        values = {variant: _long_values(rows, condition, field) for variant, rows in data.items()}
        task = "long_context_" + condition + "_full_8192"
        condition_points, condition_deltas, condition_macro, counts, complete_case_macro = _long_condition_analysis(
            values, task=task, metric=metric, transform=transform, reps=reps, rng=rng)
        points.extend(condition_points); deltas.extend(condition_deltas)
        condition_info[condition] = {"status": "complete", "matched_groups_by_language_position": counts,
                                     "point_by_variant": condition_macro}
        if condition == "intact":
            primary_point = condition_macro
            primary_summary = {"status": "complete",
                "primary_endpoint": "intact/full/8192 F1; equal en/zh and position weighting",
                "primary_by_variant": condition_macro,
                "complete_case_sensitivity_point_by_variant": complete_case_macro,
                "matched_source_groups_by_language_position": counts,
                "note": "Primary means use each language-position cell's matched groups, then equal-weight cells. Bootstrap resamples each language's source-group union once and shares draws across positions and variants. Complete-case means use only groups present at all three positions as sensitivity. Evidence-deleted is original-answer reproduction, not refusal accuracy; conditions are kept separate."}
    primary_summary["secondary_conditions"] = {k: v for k, v in condition_info.items() if k != "intact"}
    return points, deltas, primary_summary, primary_point, _long_breakdown(data)


def _parse_cost_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = _read_json(path)
    rows = data.get("cells", data.get("results"))
    if rows is None and data.get("task") == "performance" and isinstance(data.get("performance"), dict):
        rows = [dict(data["performance"], variant=data.get("variant"))]
    if rows is None and data.get("variant") and (data.get("length") or data.get("sequence_length")):
        rows = [data]
    if not isinstance(rows, list):
        raise ValueError(f"encoder cost JSON needs cells/results array or a performance result: {path}")
    normalized = []
    for cell in rows:
        if not isinstance(cell, dict):
            raise ValueError(f"invalid cost cell in {path}")
        variant = str(cell.get("variant") or data.get("variant") or "")
        length = int(cell.get("length", cell.get("sequence_length", 0)))
        batch = int(cell.get("physical_batch", cell.get("physical_batch_size", 0)))
        if variant not in TASK_VARIANTS["qa"] or length not in {512, 4096, 8192}:
            raise ValueError(f"invalid cost variant/length in {path}: {variant}/{length}")
        if batch != {512: 16, 4096: 4, 8192: 2}[length]:
            raise ValueError(f"unexpected physical batch in {path}: {length}/{batch}")
        rounds = cell.get("rounds")
        if not isinstance(rounds, list):
            perf = cell.get("performance") if isinstance(cell.get("performance"), dict) else cell
            rounds = [perf]
        for round_row in rounds:
            if not isinstance(round_row, dict):
                raise ValueError(f"invalid benchmark round in {path}")
            summary = round_row.get("summary") if isinstance(round_row.get("summary"), dict) else round_row
            memory = round_row.get("peak_memory", round_row.get("memory_peak", {}))
            normalized.append({
                "variant": variant, "length": length, "physical_batch": batch,
                "cuda_latency_p50_s": float(summary["cuda_event_encode_ms_p50"]) / 1000.0 if summary.get("cuda_event_encode_ms_p50") is not None else None,
                "cuda_latency_p95_s": float(summary["cuda_event_encode_ms_p95"]) / 1000.0 if summary.get("cuda_event_encode_ms_p95") is not None else None,
                "wall_latency_p50_s": float(summary["wall_prepared_cpu_gpu_mask_encode_ms_p50"]) / 1000.0 if summary.get("wall_prepared_cpu_gpu_mask_encode_ms_p50") is not None else None,
                "wall_latency_p95_s": float(summary["wall_prepared_cpu_gpu_mask_encode_ms_p95"]) / 1000.0 if summary.get("wall_prepared_cpu_gpu_mask_encode_ms_p95") is not None else None,
                "tokens_per_second": float(summary["nonpad_tokens_per_second"]) if summary.get("nonpad_tokens_per_second") is not None else None,
                "peak_allocated_bytes": int(memory["max_allocated_bytes"]) if memory.get("max_allocated_bytes") is not None else None,
                "peak_reserved_bytes": int(memory["max_reserved_bytes"]) if memory.get("max_reserved_bytes") is not None else None,
                "forward_count": int(summary.get("forward_count", len(round_row.get("samples", [])))),
                "source": str(path), "round": round_row.get("round"),
            })
    metadata = {"protocol": data.get("protocol"), "hardware": data.get("hardware"),
                "software": data.get("software"), "provenance": data.get("provenance"),
                "model_loads": data.get("model_loads"),
                "whole_run_gpu_memory_peak": data.get("whole_run_gpu_memory_peak"), "source": str(path)}
    return normalized, metadata


def load_cost(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not paths:
        return [], {"status": "pending_cost_json"}
    parsed = [_parse_cost_file(path) for path in paths]
    raw = [row for rows, _ in parsed for row in rows]
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        grouped[(row["variant"], row["length"])].append(row)
    expected = {(v, length) for v in TASK_VARIANTS["qa"] for length in (512, 4096, 8192)}
    if set(grouped) != expected:
        raise ValueError(f"incomplete encoder cost matrix: missing={sorted(expected-set(grouped))}, extra={sorted(set(grouped)-expected)}")
    cells = []
    for (variant, length), rows in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1])):
        first = rows[0]
        record = {"variant": variant, "length": length, "physical_batch": first["physical_batch"], "rounds": len(rows)}
        for key in ("cuda_latency_p50_s", "cuda_latency_p95_s", "wall_latency_p50_s", "wall_latency_p95_s", "tokens_per_second"):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            record[key] = float(statistics.median(values)) if values else None
            record[key + "_round_min"] = min(values) if values else None
            record[key + "_round_max"] = max(values) if values else None
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            values = [int(row[key]) for row in rows if row.get(key) is not None]
            record[key] = max(values) if values else None
        record["forward_counts_by_round"] = [r["forward_count"] for r in rows]
        record["sources"] = sorted(set(row["source"] for row in rows))
        cells.append(record)
    if any(row["rounds"] != 3 for row in cells):
        raise ValueError("encoder cost protocol expects exactly three rounds per variant/length cell")
    source_metadata = [metadata for _, metadata in parsed]
    model_load_peak: dict[str, dict[str, int]] = {}
    for metadata in source_metadata:
        for load in metadata.get("model_loads") or []:
            variant = str(load.get("variant") or "")
            peak = load.get("model_gpu_load_peak") or {}
            if not variant or not peak:
                continue
            target = model_load_peak.setdefault(variant, {"max_allocated_bytes": 0, "max_reserved_bytes": 0})
            for key in target:
                target[key] = max(target[key], int(peak.get(key, 0)))
    whole_run_peaks = [m.get("whole_run_gpu_memory_peak") for m in source_metadata if m.get("whole_run_gpu_memory_peak")]
    whole_run_peak = {key: max(int(row.get(key, 0)) for row in whole_run_peaks)
                      for key in ("max_allocated_bytes", "max_reserved_bytes")} if whole_run_peaks else None
    return cells, {"status": "complete", "cells": len(cells), "rounds_per_cell": 3,
                   "steady_state_memory_peak_scope": "max across all measured rounds in each cell",
                   "model_load_peak_by_variant": model_load_peak, "whole_run_gpu_memory_peak": whole_run_peak,
                   "source_metadata": source_metadata,
                   "timing_scopes": "reported source fields; no model execution performed by this script"}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def make_plots(outdir: Path, cost: Sequence[Mapping[str, Any]], long_quality: Mapping[str, Mapping[str, Any]]) -> list[str]:
    if not cost:
        return []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plots = []
    lengths = (512, 4096, 8192)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    for variant in TASK_VARIANTS["qa"]:
        rows = sorted((r for r in cost if r["variant"] == variant), key=lambda r: r["length"])
        xs = [int(r["length"]) for r in rows]
        ys = [r["tokens_per_second"] for r in rows]
        if all(v is not None for v in ys):
            ax.plot(xs, ys, marker="o", label=variant)
    ax.set_xscale("log", base=2); ax.set_xticks(lengths, labels=[f"{x}\nB={ {512: 16, 4096: 4, 8192: 2}[x] }" for x in lengths])
    ax.set_xlabel("Input length (tokens)"); ax.set_ylabel("Nonpadding tokens / second")
    ax.set_title("Encoder throughput by input length"); ax.grid(True, alpha=0.3); ax.legend()
    path = outdir / "encoder_efficiency_by_length.png"; fig.savefig(path, dpi=160); plt.close(fig); plots.append(str(path))
    if long_quality and all(any(r["variant"] == v and r["length"] == 8192 and r["cuda_latency_p50_s"] is not None for r in cost) for v in long_quality):
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        for variant, quality_row in sorted(long_quality.items()):
            row = next(r for r in cost if r["variant"] == variant and r["length"] == 8192)
            quality = float(quality_row["estimate"])
            yerr = [[max(0.0, quality-float(quality_row["ci_low"]))],
                    [max(0.0, float(quality_row["ci_high"])-quality)]]
            ax.errorbar(row["cuda_latency_p50_s"], quality, yerr=yerr, fmt="o", label=variant, capsize=3, markersize=7)
            ax.annotate(variant, (row["cuda_latency_p50_s"], quality), xytext=(5, 5), textcoords="offset points")
        ax.set_xlabel("8K encoder-forward latency p50 (CUDA events; not task end-to-end)")
        ax.set_ylabel("Intact/full/8K QA F1 (equal language and evidence position)")
        ax.set_title("8K quality–latency tradeoff"); ax.grid(True, alpha=0.3); ax.legend()
        path = outdir / "quality_cost_8k.png"; fig.savefig(path, dpi=160); plt.close(fig); plots.append(str(path))
    return plots


def _parse_long_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("long JSON must be VARIANT=PATH")
    variant, filename = value.split("=", 1)
    if variant not in TASK_VARIANTS["qa"]:
        raise argparse.ArgumentTypeError(f"invalid variant: {variant}")
    return variant, Path(filename)


def main(argv: Sequence[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=root / "results/downstream_evaluation/nextbert-formal-results.tar.gz")
    parser.add_argument("--summary", type=Path, default=root / "results/downstream_evaluation/summary.json")
    parser.add_argument("--metrics-csv", type=Path, default=root / "results/downstream_evaluation/metrics.csv")
    parser.add_argument("--bm25-baseline", type=Path, default=root / "results/downstream_evaluation/bm25_baseline.json")
    parser.add_argument("--qa-data-dir", type=Path, default=root / "data/prepared/full_67m")
    parser.add_argument("--reranking-data-dir", type=Path, default=root / "data/prepared/full_67m")
    parser.add_argument("--outdir", type=Path, default=root / "results/downstream_evaluation/quality_cost_analysis")
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cost-json", type=Path, action="append", default=[])
    parser.add_argument("--long-json", type=_parse_long_arg, action="append", default=[])
    args = parser.parse_args(argv)
    if args.bootstrap < 100:
        parser.error("--bootstrap must be at least 100")
    splits = ("dev", "test") if args.split == "all" else (args.split,)
    summary = _read_json(args.summary)
    if summary.get("status") != "complete":
        raise ValueError("downstream summary is not complete")
    archive_results = load_result_archive(args.archive)
    summary_checks = _validate_summary(summary, archive_results)
    metric_rows, delta_rows, notes = analyze_saved_tasks(
        archive_results, summary, args.qa_data_dir, args.reranking_data_dir, args.bm25_baseline,
        reps=args.bootstrap, seed=args.seed, splits=splits)
    csv_check = _csv_crosscheck(args.metrics_csv, metric_rows)
    long_map = dict(args.long_json)
    if len(long_map) != len(args.long_json):
        raise ValueError("duplicate --long-json variant")
    long_metrics, long_deltas, long_summary, long_primary, long_breakdown = analyze_long_context(long_map, reps=args.bootstrap, seed=args.seed)
    metric_rows.extend(long_metrics); delta_rows.extend(long_deltas)
    cost_cells, cost_summary = load_cost(args.cost_json)
    primary_plot_rows = {row["variant"]: row for row in long_metrics
        if row["task"] == "long_context_intact_full_8192" and row["language"] == "en_zh_macro"
        and row["metric"] == "F1_equal_position"}
    plots = make_plots(args.outdir, cost_cells, primary_plot_rows)
    out = {
        "schema_version": 1,
        "status": "complete",
        "analysis": {"split": args.split, "bootstrap_replicates": args.bootstrap, "confidence_level": 0.95,
                     "bootstrap_seed": args.seed, "training_seeds_added": 0, "language_aggregation": "equal en/zh macro",
                     "quality_cost_scalar": False},
        "provenance": {"archive": str(args.archive), "summary": str(args.summary),
                       "metrics_csv": str(args.metrics_csv), "summary_prediction_metrics_checked": summary_checks,
                       "metrics_csv_point_rows_checked": csv_check["rows_checked"],
                       "metrics_csv_validation": csv_check["status"]},
        "task_metrics": metric_rows,
        "paired_deltas": delta_rows,
        "long_context": long_summary,
        "long_breakdown": long_breakdown,
        "encoder_cost": {**cost_summary, "cells": cost_cells},
        "plots": plots,
        "notes": notes + [
            "Retrieval query-level bootstrap is unavailable when saved combined outputs omit per-query predictions.",
            "Quality deltas are descriptive paired bootstrap intervals; they are not additional training seeds or a scalar quality score.",
        ],
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "quality_cost_summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_csv(args.outdir / "task_metrics.csv", metric_rows)
    _write_csv(args.outdir / "paired_deltas.csv", delta_rows)
    if cost_cells:
        _write_csv(args.outdir / "encoder_cost.csv", cost_cells)
    if long_breakdown:
        _write_csv(args.outdir / "long_breakdown.csv", long_breakdown)
        (args.outdir / "long_breakdown.json").write_text(json.dumps(long_breakdown, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "outdir": str(args.outdir),
                      "task_metric_rows": len(metric_rows), "paired_delta_rows": len(delta_rows),
                      "summary_prediction_metrics_checked": summary_checks,
                      "metrics_csv_point_rows_checked": csv_check["rows_checked"],
                      "cost_status": cost_summary["status"], "long_status": long_summary["status"],
                      "plots": plots}, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
