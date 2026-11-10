"""Offline evaluation and real-data cost probes for NextBERT checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from train_nextbert import (
    PACKAGE_ROOT, _amp_context, _device, _embeddings, _field, _logits,
    _memory_stats, _method, _seed_everything, _special_ids, collate_sequences,
    load_jsonl, load_nextbert, pair_tokens,
)


def _load_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload.get("model", payload))


def _content_mask(model: torch.nn.Module, input_ids: torch.Tensor,
                  attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    for special in getattr(model, "special_token_ids", None) or _special_ids():
        mask &= input_ids.ne(int(special))
    return mask


def _to_int_ids(value: Any) -> list[int]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("token ID fields must be nonempty arrays")
    return [int(token) for token in value]


def _language(row: Mapping[str, Any]) -> str:
    return str(row.get("language", "und"))


def _metric_by_language(predictions: Sequence[Mapping[str, Any]], metric: str) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[_language(row)].append(row)
    result: dict[str, Any] = {}
    for language, rows in sorted(groups.items()):
        result[language] = {metric: sum(bool(row[metric]) for row in rows) / len(rows),
                            "count": len(rows)}
    return result


def evaluate_classification(model: torch.nn.Module, rows: Sequence[Mapping[str, Any]], *,
                            device: torch.device, batch_size: int = 32) -> dict[str, Any]:
    predictions: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            ids, attention = collate_sequences([_to_int_ids(row["input_ids"]) for row in batch],
                                               pad_token_id=0, device=device)
            content = _content_mask(model, ids, attention)
            with _amp_context(device):
                output = _method(model, "forward_classification")(
                    input_ids=ids, attention_mask=attention, content_mask=content)
            logits = _logits(output).float()
            predicted = logits.argmax(dim=-1).cpu().tolist()
            for row, prediction in zip(batch, predicted):
                label = int(row["label"])
                predictions.append({"id": row.get("id"), "language": _language(row),
                                    "prediction": int(prediction), "label": label,
                                    "correct": int(prediction) == label})
    accuracy = sum(row["correct"] for row in predictions) / max(1, len(predictions))
    return {"count": len(predictions), "accuracy": accuracy,
            "macro_f1": _macro_f1(predictions),
            "by_language": _classification_by_language(predictions),
            "predictions": predictions}


def _macro_f1(rows: Sequence[Mapping[str, Any]], labels: Sequence[int] = (0, 1, 2)) -> float:
    scores = []
    for label in labels:
        true_positive = sum(int(row["prediction"] == label and row["label"] == label) for row in rows)
        false_positive = sum(int(row["prediction"] == label and row["label"] != label) for row in rows)
        false_negative = sum(int(row["prediction"] != label and row["label"] == label) for row in rows)
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return statistics.mean(scores)


def _classification_by_language(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_language(row)].append(row)
    return {language: {
        "count": len(group),
        "accuracy": sum(row["correct"] for row in group) / len(group),
        "macro_f1": _macro_f1(group),
    } for language, group in sorted(grouped.items())}


def _best_span(start_logits: torch.Tensor, end_logits: torch.Tensor,
               context_mask: torch.Tensor, max_answer_length: int = 30) -> tuple[int, int, float]:
    context_valid = context_mask.bool()
    length = int(context_valid.numel())
    if length == 0:
        raise ValueError("QA context_mask contains no valid context tokens")
    width = min(max_answer_length, length)
    starts = torch.arange(length, device=start_logits.device).unsqueeze(1)
    offsets = torch.arange(width, device=start_logits.device).unsqueeze(0)
    ends = starts + offsets
    in_bounds = ends < length
    safe_ends = ends.clamp_max(length - 1)
    invalid_prefix = (~context_valid).to(torch.int32).cumsum(dim=0)
    prefix_before_start = torch.cat((invalid_prefix.new_zeros(1), invalid_prefix[:-1]))
    invalid_inside = invalid_prefix[safe_ends] - prefix_before_start.unsqueeze(1)
    valid = (in_bounds & context_valid.unsqueeze(1) & context_valid[safe_ends]
             & invalid_inside.eq(0))
    scores = start_logits.float().unsqueeze(1) + end_logits.float()[safe_ends]
    scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
    flat_scores = scores.reshape(-1)
    flat_index = flat_scores.argmax()
    best_score = flat_scores[flat_index]
    selected = torch.stack((flat_index.float(), best_score)).detach().cpu().tolist()
    flat, score = int(selected[0]), float(selected[1])
    return flat // width, flat // width + flat % width, score


def _gold_texts(row: Mapping[str, Any]) -> list[str]:
    answers = row.get("answers", row.get("answer_texts", row.get("answer_text", [])))
    if isinstance(answers, Mapping):
        values = answers.get("text", [])
    elif isinstance(answers, str):
        values = [answers]
    else:
        values = answers
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values or []]


def _predicted_answer_text(row: Mapping[str, Any], token_ids: Sequence[int],
                           start: int, end: int) -> str:
    context = row.get("context_text", row.get("context"))
    offsets = row.get("offset_mapping", row.get("offsets"))
    if isinstance(context, str) and isinstance(offsets, (list, tuple)) and end < len(offsets):
        try:
            start_offset = offsets[start]
            end_offset = offsets[end]
            if isinstance(start_offset, Mapping):
                char_start = int(start_offset.get("start", 0))
            else:
                char_start = int(start_offset[0])
            if isinstance(end_offset, Mapping):
                char_end = int(end_offset.get("end", 0))
            else:
                char_end = int(end_offset[1])
            if char_end >= char_start:
                return context[char_start:char_end]
        except (IndexError, TypeError, ValueError):
            pass
    return " ".join(str(token) for token in token_ids[start:end + 1])


def evaluate_qa(model: torch.nn.Module, rows: Sequence[Mapping[str, Any]], *,
                 device: torch.device, batch_size: int = 8) -> dict[str, Any]:
    try:
        from qa_metrics import answer_em, answer_f1
    except ImportError as exc:
        raise RuntimeError("official QA scoring requires src/qa_metrics.py") from exc
    by_question: dict[tuple[str, str, str], dict[str, Any]] = {}
    model.eval()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            sequences = [_to_int_ids(row["input_ids"]) for row in batch]
            ids, attention = collate_sequences(sequences, pad_token_id=0, device=device)
            contexts = [row.get("context_mask", [1] * len(row["input_ids"])) for row in batch]
            if any(not any(int(value) for value in context) for context in contexts):
                raise ValueError("QA context_mask contains a window with no context tokens")
            context_ids, _ = collate_sequences(contexts, pad_token_id=0, device=device)
            context_mask = context_ids.bool() & attention.bool()
            with _amp_context(device):
                output = _method(model, "forward_qa")(
                    input_ids=ids, attention_mask=attention, context_mask=context_mask)
            start_logits = _field(output, "start_logits")
            end_logits = _field(output, "end_logits")
            if not isinstance(start_logits, torch.Tensor) or not isinstance(end_logits, torch.Tensor):
                logits = _logits(output, ("logits",))
                start_logits, end_logits = logits[..., 0], logits[..., 1]
            for index, row in enumerate(batch):
                predicted_start, predicted_end, score = _best_span(
                    start_logits[index], end_logits[index], context_mask[index])
                tokens = sequences[index]
                question_id = str(row.get("question_id", row.get("id", len(by_question))))
                dataset = str(row.get("dataset", row.get("source", "")))
                group_key = (_language(row), dataset, question_id)
                candidate = {
                    "id": question_id, "language": _language(row),
                    "dataset": dataset,
                    "prediction_start": predicted_start, "prediction_end": predicted_end,
                    "prediction_score": score,
                    "prediction_text": _predicted_answer_text(row, tokens, predicted_start, predicted_end),
                    "gold_texts": _gold_texts(row),
                }
                previous = by_question.get(group_key)
                # Pick the highest-scoring model span across every overflow
                # window. Gold positions never participate in window selection.
                if previous is None or score > previous["prediction_score"]:
                    by_question[group_key] = candidate
                elif not previous["gold_texts"] and candidate["gold_texts"]:
                    previous["gold_texts"] = candidate["gold_texts"]
    predictions: list[dict[str, Any]] = []
    for (language, dataset, question_id), row in by_question.items():
        gold_texts = row["gold_texts"]
        if not gold_texts:
            # Fallback for compact prepared files that retain token span labels
            # but omit raw text. Invalid (-100) windows do not become answers.
            matching = [item for item in rows
                        if _language(item) == language
                        and str(item.get("dataset", item.get("source", ""))) == dataset
                        and str(item.get("question_id", item.get("id", ""))) == question_id
                        and int(item.get("start_positions", -100)) >= 0
                        and int(item.get("end_positions", -100)) >= 0]
            if matching:
                source = matching[0]
                ids = _to_int_ids(source["input_ids"])
                gold_texts = [" ".join(str(token) for token in ids[
                    int(source["start_positions"]):int(source["end_positions"]) + 1])]
        exact = max((float(answer_em(row["prediction_text"], gold, row["language"]))
                     for gold in gold_texts), default=0.0)
        f1 = max((float(answer_f1(row["prediction_text"], gold, row["language"]))
                  for gold in gold_texts), default=0.0)
        predictions.append({key: value for key, value in row.items() if key != "gold_texts"}
                           | {"exact_match": exact, "f1": f1})
    return {"count": len(predictions),
            "overflow_windows": len(rows),
            "question_count": len(predictions),
            "scoring_status": "official SQuAD/CMRC scorer via qa_metrics.answer_em/answer_f1",
            "exact_match": statistics.mean([row["exact_match"] for row in predictions]) if predictions else 0.0,
            "f1": statistics.mean([row["f1"] for row in predictions]) if predictions else 0.0,
            "exact_match_by_language": _metric_by_language(predictions, "exact_match"),
            "f1_by_language": _metric_by_language(predictions, "f1"),
            "predictions": predictions}


def _doc_ids(row: Mapping[str, Any]) -> str:
    return str(row.get("id", row.get("document_id", "")))


def _bucket(length: int) -> str:
    if length <= 512:
        return "up_to_512"
    if length <= 4096:
        return "513_to_4096"
    return "4097_to_8192"


def _stable_sample(rows: Sequence[Mapping[str, Any]], limit: int,
                   required_ids: set[str] | None = None) -> tuple[list[Mapping[str, Any]], list[str]]:
    required_ids = required_ids or set()
    required = sorted((row for row in rows if _doc_ids(row) in required_ids),
                      key=lambda row: hashlib.sha256(_doc_ids(row).encode()).digest())
    # Preserve some known positives, while reserving most slots for a
    # length-stratified timing sample so every populated bucket is measured.
    required_limit = min(len(required), max(0, limit // 4))
    selected = list(required[:required_limit])
    selected_ids = {_doc_ids(row) for row in selected}
    remaining = [row for row in rows if _doc_ids(row) not in selected_ids]
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in remaining:
        grouped[_bucket(min(8192, len(row["input_ids"])))].append(row)
    for bucket_rows in grouped.values():
        bucket_rows.sort(key=lambda row: hashlib.sha256(_doc_ids(row).encode()).digest())
    available = max(0, limit - len(selected))
    quotas = _allocate_bucket_counts({bucket: len(values) for bucket, values in grouped.items()}, available)
    for bucket, bucket_rows in grouped.items():
        count = min(quotas.get(bucket, 0), len(bucket_rows))
        selected.extend(bucket_rows[:count])
    if len(selected) < limit:
        selected_ids = {_doc_ids(row) for row in selected}
        leftovers = [row for row in rows if _doc_ids(row) not in selected_ids]
        leftovers.sort(key=lambda row: hashlib.sha256(_doc_ids(row).encode()).digest())
        selected.extend(leftovers[:limit - len(selected)])
    missing = sorted(required_ids - {_doc_ids(row) for row in selected})
    return selected[:limit], missing


def _encode_doc_batches(model: torch.nn.Module, rows: Sequence[Mapping[str, Any]], *,
                        device: torch.device, bucket_batch_size: Mapping[str, int],
                        measure: bool = True) -> tuple[np.ndarray, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        ids = _to_int_ids(row["input_ids"])
        if len(ids) > 8192:
            ids = ids[:8192]
        grouped[_bucket(len(ids))].append({**row, "_effective_ids": ids})
    vectors: dict[str, torch.Tensor] = {}
    bucket_stats: dict[str, Any] = {}
    model.eval()
    for bucket, bucket_rows in grouped.items():
        batch_size = int(bucket_batch_size[bucket])
        bucket_vectors: list[torch.Tensor] = []
        started = time.perf_counter()
        batch_times: list[float] = []
        token_count = 0
        for offset in range(0, len(bucket_rows), batch_size):
            batch = bucket_rows[offset:offset + batch_size]
            ids, attention = collate_sequences([row["_effective_ids"] for row in batch],
                                               pad_token_id=0, device=device)
            content = _content_mask(model, ids, attention)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_started = time.perf_counter()
            with torch.inference_mode(), _amp_context(device):
                encoded = _embeddings(_method(model, "encode_retrieval")(
                    input_ids=ids, attention_mask=attention, content_mask=content))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_times.append(time.perf_counter() - batch_started)
            bucket_vectors.append(encoded.float().cpu())
            token_count += int(attention.sum().item())
        elapsed = time.perf_counter() - started if measure else sum(batch_times)
        vectors[bucket] = torch.cat(bucket_vectors, dim=0) if bucket_vectors else torch.empty((0, 256))
        bucket_stats[bucket] = {
            "documents": len(bucket_rows), "nonpad_tokens": token_count,
            "wall_seconds_including_collation_and_transfer": elapsed,
            "model_forward_batch_seconds": sum(batch_times),
            "mean_forward_seconds_per_document": sum(batch_times) / max(1, len(bucket_rows)),
            "mean_index_pipeline_seconds_per_document": elapsed / max(1, len(bucket_rows)),
            "mean_tokens_per_document": token_count / max(1, len(bucket_rows)),
            "physical_document_batch": batch_size,
        }
    ordered = []
    for bucket, bucket_rows in grouped.items():
        ordered.extend((row, vector) for row, vector in zip(bucket_rows, vectors[bucket]))
    output = np.zeros((len(ordered), int(ordered[0][1].numel()) if ordered else 256), dtype=np.float32)
    ordered_rows: list[dict[str, Any]] = []
    for index, (row, vector) in enumerate(ordered):
        output[index] = vector.numpy()
        ordered_rows.append({key: value for key, value in row.items() if key != "_effective_ids"})
    return output, {"bucket_stats": bucket_stats, "ordered_rows": ordered_rows}


def _relevance_ids(row: Mapping[str, Any]) -> set[str]:
    values = row.get("relevant_doc_ids", row.get("relevant_ids", row.get("positive_doc_ids", [])))
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def _dcg(relevances: Sequence[float], cutoff: int) -> float:
    return sum((2.0 ** float(relevance) - 1.0) / math.log2(rank + 2)
               for rank, relevance in enumerate(relevances[:cutoff]))


def _retrieval_metrics(ranked_ids: Sequence[str], relevant: set[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for cutoff in (10, 100):
        result[f"Recall@{cutoff}"] = len(set(ranked_ids[:cutoff]) & relevant) / max(1, len(relevant))
    rel10 = [1.0 if document in relevant else 0.0 for document in ranked_ids[:10]]
    ideal = [1.0] * min(10, len(relevant))
    result["nDCG@10"] = _dcg(rel10, 10) / max(_dcg(ideal, 10), 1e-12)
    return result


def evaluate_retrieval(model: torch.nn.Module, corpus: Sequence[Mapping[str, Any]],
                       queries: Sequence[Mapping[str, Any]], *, device: torch.device,
                       sample_documents_per_language: int | None = None,
                       document_batch_size: int | None = None,
                       store_index: Path | None = None) -> dict[str, Any]:
    corpus_by_language: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    queries_by_language: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in corpus:
        corpus_by_language[_language(row)].append(row)
    for row in queries:
        queries_by_language[_language(row)].append(row)
    all_results: list[dict[str, Any]] = []
    index_projection: dict[str, Any] = {}
    for language, language_corpus in sorted(corpus_by_language.items()):
        language_queries = queries_by_language.get(language, [])
        relevant = set().union(*(_relevance_ids(row) for row in language_queries)) if language_queries else set()
        required = relevant if sample_documents_per_language else set()
        if sample_documents_per_language:
            selected, missing_required = _stable_sample(language_corpus, sample_documents_per_language, required)
        else:
            selected, missing_required = list(language_corpus), []
        if not selected:
            continue
        if document_batch_size is not None:
            batch_by_bucket = {name: document_batch_size for name in
                               ("up_to_512", "513_to_4096", "4097_to_8192")}
        else:
            batch_by_bucket = {"up_to_512": 4, "513_to_4096": 4, "4097_to_8192": 2}
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        index_started = time.perf_counter()
        vectors, encode_info = _encode_doc_batches(model, selected, device=device,
                                                   bucket_batch_size=batch_by_bucket)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        index_seconds = time.perf_counter() - index_started
        doc_ids = [_doc_ids(row) for row in encode_info["ordered_rows"]]
        id_positions = {doc_id: position for position, doc_id in enumerate(doc_ids)}
        query_predictions: list[dict[str, Any]] = []
        query_encode_times: list[float] = []
        model.eval()
        for query in language_queries:
            query_token_ids = query.get("query_ids", query.get("input_ids"))
            if query_token_ids is None:
                raise ValueError("retrieval query row must contain query_ids or input_ids")
            query_ids = _to_int_ids(query_token_ids)
            ids, attention = collate_sequences([query_ids], pad_token_id=0, device=device)
            content = _content_mask(model, ids, attention)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            query_started = time.perf_counter()
            with torch.inference_mode(), _amp_context(device):
                query_vector = _embeddings(_method(model, "encode_retrieval")(
                    input_ids=ids, attention_mask=attention, content_mask=content)).float().cpu().numpy()[0]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            query_encode_times.append(time.perf_counter() - query_started)
            scores = vectors @ query_vector
            order = np.argsort(-scores)
            ranked = [doc_ids[int(index)] for index in order]
            qrels = _relevance_ids(query)
            present_qrels = qrels.intersection(id_positions)
            metrics = _retrieval_metrics(ranked, present_qrels) if present_qrels else {
                "nDCG@10": 0.0, "Recall@10": 0.0, "Recall@100": 0.0}
            query_predictions.append({"id": query.get("id"), "language": language,
                                     "metrics": metrics, "relevant_present": len(present_qrels),
                                     "relevant_total": len(qrels)})
        metric_means = {
            name: statistics.mean([row["metrics"][name] for row in query_predictions])
            if query_predictions else None
            for name in ("nDCG@10", "Recall@10", "Recall@100")
        }
        bucket_counts = {name: 0 for name in batch_by_bucket}
        for row in language_corpus:
            bucket_counts[_bucket(len(row["input_ids"]))] += 1
        target_counts = _allocate_bucket_counts(bucket_counts, 5000)
        projected_seconds = 0.0
        projected_known = True
        for bucket, count in target_counts.items():
            stat = encode_info["bucket_stats"].get(bucket)
            if count and not stat:
                projected_known = False
                continue
            if stat:
                projected_seconds += float(stat["mean_index_pipeline_seconds_per_document"]) * count
        language_result = {
            "language": language, "corpus_documents_available": len(language_corpus),
            "documents_indexed": len(selected), "queries_evaluated": len(query_predictions),
            "sampled_index_only": sample_documents_per_language is not None,
            "sample_documents_per_language": sample_documents_per_language,
            "known_relevant_docs_missing_from_sample": missing_required,
            "retrieval_metrics_on_this_candidate_pool": metric_means,
            "query_encoding_seconds_mean": statistics.mean(query_encode_times) if query_encode_times else 0.0,
            "query_encoding_seconds_p50": float(np.percentile(query_encode_times, 50)) if query_encode_times else None,
            "query_encoding_seconds_p95": float(np.percentile(query_encode_times, 95)) if query_encode_times else None,
            "index_build_seconds": index_seconds,
            "index_encoding_by_length_bucket": encode_info["bucket_stats"],
            "vector_storage_bytes": int(vectors.nbytes),
            "vector_storage_mib": vectors.nbytes / (1024 ** 2),
            "projected_5000_vector_storage_bytes":
                int(vectors.shape[1] * 4 * 5000) if vectors.ndim == 2 else None,
            "projected_5000_document_index_seconds_from_bucket_rates":
                projected_seconds if projected_known else None,
            "projected_5000_document_scope": "single language, measured bucket rates, excludes disk serialization",
            "device_memory_after_index": _memory_stats(device),
            "query_predictions": query_predictions,
        }
        if store_index is not None:
            store_index.parent.mkdir(parents=True, exist_ok=True)
            np.save(store_index.with_name(f"{store_index.stem}_{language}.npy"), vectors)
            language_result["stored_vector_file"] = str(store_index.with_name(f"{store_index.stem}_{language}.npy"))
        all_results.append(language_result)
        index_projection[language] = projected_seconds if projected_known else None
    return {"languages": all_results,
            "total_index_build_seconds": sum(row["index_build_seconds"] for row in all_results),
            "projected_5000_per_language_index_seconds": index_projection,
            "calibration_only_when_sampled": sample_documents_per_language is not None}


def _allocate_bucket_counts(observed: Mapping[str, int], target: int) -> dict[str, int]:
    total = sum(observed.values())
    if total == 0:
        return {bucket: 0 for bucket in observed}
    exact = {bucket: target * count / total for bucket, count in observed.items()}
    allocated = {bucket: int(math.floor(value)) for bucket, value in exact.items()}
    remainder = target - sum(allocated.values())
    for bucket in sorted(exact, key=lambda key: exact[key] - allocated[key], reverse=True)[:remainder]:
        allocated[bucket] += 1
    return allocated


def _candidate_documents(row: Mapping[str, Any],
                         corpus_by_id: Mapping[str, Mapping[str, Any]] | None = None
                         ) -> tuple[list[list[int]], list[int], list[str]]:
    if "candidate_ids" in row:
        raw_candidates = row["candidate_ids"]
        if not isinstance(raw_candidates, (list, tuple)) or not raw_candidates:
            raise ValueError("candidate_ids must be a nonempty list")
        first = raw_candidates[0]
        if isinstance(first, (int, np.integer)):
            token_candidates = [raw_candidates]
        elif isinstance(first, (list, tuple, np.ndarray, torch.Tensor)):
            token_candidates = raw_candidates
        else:
            token_candidates = None

        supplied_docids = row.get("candidate_docids")
        if token_candidates is not None:
            if supplied_docids is not None:
                candidate_ids = [str(value) for value in supplied_docids]
            else:
                candidate_ids = [str(index) for index in range(len(token_candidates))]
            if len(candidate_ids) != len(token_candidates):
                raise ValueError("candidate_docids and token candidate lists must have equal lengths")
            if corpus_by_id is not None and supplied_docids is not None:
                missing = [value for value in candidate_ids if value not in corpus_by_id]
                if missing:
                    raise ValueError(f"reranking candidates are missing from corpus: {missing[:3]}")
                docs = [_to_int_ids(corpus_by_id[value]["input_ids"]) for value in candidate_ids]
            else:
                docs = [_to_int_ids(value) for value in token_candidates]
        else:
            candidate_ids = ([str(value) for value in supplied_docids]
                             if supplied_docids is not None
                             else [str(value) for value in raw_candidates])
            if corpus_by_id is None:
                raise ValueError("document-ID reranking rows require --corpus")
            missing = [value for value in candidate_ids if value not in corpus_by_id]
            if missing:
                raise ValueError(f"reranking candidates are missing from corpus: {missing[:3]}")
            docs = [_to_int_ids(corpus_by_id[value]["input_ids"]) for value in candidate_ids]
        labels = [int(value) for value in row.get("labels", [0] * len(candidate_ids))]
        if len(labels) != len(docs):
            raise ValueError("reranking labels and candidate_ids must have equal lengths")
        return docs, labels, candidate_ids
    candidates = row.get("candidates", row.get("documents"))
    if candidates:
        docs: list[list[int]] = []
        labels: list[int] = []
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                token_ids = candidate.get("input_ids", candidate.get("document_ids", candidate.get("ids")))
                relevance = candidate.get("relevance", candidate.get("label", candidate.get("relevant", 0)))
                labels.append(int(bool(relevance)) if not isinstance(relevance, (int, float)) else int(relevance))
            else:
                token_ids = candidate
                labels.append(0)
            docs.append(_to_int_ids(token_ids))
        return docs, labels, [str(candidate.get("id", index)) if isinstance(candidate, Mapping)
                              else str(index) for index, candidate in enumerate(candidates)]
    if "positive_ids" in row and "negative_ids" in row:
        positives = row["positive_ids"]
        negatives = row["negative_ids"]
        if positives and isinstance(positives[0], (int, np.integer)):
            positives = [positives]
        if negatives and isinstance(negatives[0], (int, np.integer)):
            negatives = [negatives]
        docs = [_to_int_ids(doc) for doc in positives] + [_to_int_ids(doc) for doc in negatives]
        return docs, [1] * len(positives) + [0] * len(negatives), [str(index) for index in range(len(docs))]
    raise ValueError("reranking row needs candidates or positive_ids/negative_ids")


def evaluate_reranking(model: torch.nn.Module, rows: Sequence[Mapping[str, Any]], *,
                       device: torch.device, sample_queries: int | None = None,
                       require_20_candidates: bool = False,
                       corpus: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    selected = list(rows)
    selected.sort(key=lambda row: hashlib.sha256(str(row.get("id", "")).encode()).digest())
    if sample_queries is not None:
        selected = selected[:sample_queries]
    model.eval()
    results: list[dict[str, Any]] = []
    pair_forward_seconds: list[float] = []
    query_seconds: list[float] = []
    query_timing_rows: list[dict[str, Any]] = []
    corpus_by_id = {_doc_ids(row): row for row in corpus or []}
    if sample_queries is not None:
        sampled_by_language: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in selected:
            sampled_by_language[_language(row)].append(row)
        selected = []
        for language_rows in sampled_by_language.values():
            selected.extend(language_rows[:sample_queries])
    for row in selected:
        docs, labels, candidate_ids = _candidate_documents(row, corpus_by_id if corpus else None)
        if require_20_candidates and len(docs) != 20:
            raise ValueError(f"reranking calibration expects exactly 20 candidates; got {len(docs)}")
        query_ids = _to_int_ids(row["query_ids"])
        pairs = [pair_tokens(query_ids, doc, sep_token_id=1, max_length=1024) for doc in docs]
        scores: list[float] = []
        query_started = time.perf_counter()
        for offset in range(0, len(pairs), 12):
            batch_pairs = pairs[offset:offset + 12]
            ids, attention = collate_sequences(batch_pairs, pad_token_id=0, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode(), _amp_context(device):
                output = _method(model, "forward_reranking")(
                    input_ids=ids, attention_mask=attention, group_size=len(pairs))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            pair_forward_seconds.append(time.perf_counter() - started)
            scores.extend(_logits(output, ("scores", "logits")).float().reshape(-1).cpu().tolist())
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        query_seconds.append(time.perf_counter() - query_started)
        query_timing_rows.append({"id": row.get("id"), "language": _language(row),
                                  "seconds": query_seconds[-1]})
        order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        ranked_labels = [labels[index] for index in order]
        relevant_total = sum(label > 0 for label in labels)
        known_relevant = _relevance_ids(row)
        if known_relevant:
            candidate_relevant = known_relevant.intersection(candidate_ids)
            candidate_recall = len(candidate_relevant) / max(1, len(known_relevant))
        else:
            candidate_recall = 1.0 if relevant_total else 0.0
        reciprocal_rank = next((1.0 / rank for rank, label in enumerate(ranked_labels[:10], 1) if label > 0), 0.0)
        ndcg = _dcg(ranked_labels, 10) / max(_dcg(sorted(labels, reverse=True), 10), 1e-12)
        results.append({"id": row.get("id"), "language": _language(row),
                        "nDCG@10": ndcg, "MRR@10": reciprocal_rank,
                        "candidate_Recall@20": candidate_recall,
                        "candidate_count": len(docs)})
    by_language = {}
    for language in sorted({_language(row) for row in results}):
        language_rows = [row for row in results if row["language"] == language]
        language_times = [row["seconds"] for row in query_timing_rows if row["language"] == language]
        by_language[language] = {
            metric: statistics.mean(row[metric] for row in language_rows)
            for metric in ("nDCG@10", "MRR@10", "candidate_Recall@20")
        } | {"queries": len(language_rows),
             "mean_query_end_to_end_seconds": statistics.mean(language_times) if language_times else 0.0,
             "query_end_to_end_seconds_p50": float(np.percentile(language_times, 50)) if language_times else None,
             "query_end_to_end_seconds_p95": float(np.percentile(language_times, 95)) if language_times else None}
    return {
        "query_count": len(results), "by_language": by_language,
        "mean_query_end_to_end_seconds": statistics.mean(query_seconds) if query_seconds else 0.0,
        "query_end_to_end_seconds_p50": float(np.percentile(query_seconds, 50)) if query_seconds else None,
        "query_end_to_end_seconds_p95": float(np.percentile(query_seconds, 95)) if query_seconds else None,
        "pair_forward_batch_seconds_p50": float(np.percentile(pair_forward_seconds, 50)) if pair_forward_seconds else None,
        "pair_forward_batch_seconds_p95": float(np.percentile(pair_forward_seconds, 95)) if pair_forward_seconds else None,
        "candidate_count_mean": statistics.mean([row["candidate_count"] for row in results]) if results else None,
        "predictions": results,
    }


def _evaluate_performance(model: torch.nn.Module, data_path: Path, *, device: torch.device,
                          length: int, warmup_steps: int, measured_steps: int) -> dict[str, Any]:
    array = np.load(data_path, mmap_mode="r", allow_pickle=False)
    if array.dtype != np.int32 or array.ndim != 2 or array.shape[1] != length:
        raise ValueError(f"performance data must be int32 [N,{length}]")
    physical = {512: 16, 4096: 4, 8192: 2}[length]
    ids_np = np.asarray(array[:physical], dtype=np.int64)
    latencies: list[float] = []
    throughputs: list[float] = []
    model.eval()
    for index in range(warmup_steps + measured_steps):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        ids = torch.from_numpy(ids_np).to(device)
        attention = ids.ne(0).long()
        with torch.inference_mode(), _amp_context(device):
            output = _method(model, "encode")(input_ids=ids, attention_mask=attention)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        if index >= warmup_steps:
            tokens = int(attention.sum().item())
            latencies.append(elapsed)
            throughputs.append(tokens / max(elapsed, 1e-9))
    return {
        "length": length, "physical_batch": physical,
        "warmup_steps": warmup_steps, "measured_steps": measured_steps,
        "forward_latency_seconds_p50": float(np.percentile(latencies, 50)),
        "forward_latency_seconds_p95": float(np.percentile(latencies, 95)),
        "nonpad_tokens_per_second_mean": statistics.mean(throughputs),
        "latencies_seconds": latencies,
        "memory_peak": _memory_stats(device),
        "end_to_end_scope": "prepared token rows copied to device, attention mask construction, and encode forward; excludes tokenization",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True,
                        choices=("classification", "qa", "retrieval", "reranking", "performance"))
    parser.add_argument("--variant", required=True, choices=("V0", "V1", "V2", "V3"))
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--corpus", type=Path,
                        help="retrieval corpus JSONL: id, language, input_ids")
    parser.add_argument("--sample-documents-per-language", type=int,
                        help="build an explicitly calibration-only, real-data index sample")
    parser.add_argument("--store-index", type=Path)
    parser.add_argument("--require-20-candidates", action="store_true")
    parser.add_argument("--sample-queries", type=int)
    parser.add_argument("--length", type=int, choices=(512, 4096, 8192))
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--output", type=Path, default=PACKAGE_ROOT / "results" / "evaluation.json")
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"local prepared data missing: {args.data}")
    device = _device(args.device)
    _seed_everything(args.seed)
    model = load_nextbert(args.model_dir, args.variant, device)
    if args.checkpoint:
        _load_checkpoint(model, args.checkpoint, device)
    rows: list[dict[str, Any]] = []
    if args.task != "performance":
        rows = load_jsonl(args.data)
    started = time.perf_counter()
    if args.task == "classification":
        result = evaluate_classification(model, rows, device=device, batch_size=args.batch_size or 32)
    elif args.task == "qa":
        result = evaluate_qa(model, rows, device=device, batch_size=args.batch_size or 8)
    elif args.task == "retrieval":
        if args.corpus is None:
            raise ValueError("retrieval evaluation requires --corpus local corpus JSONL")
        corpus = load_jsonl(args.corpus)
        result = evaluate_retrieval(
            model, corpus, rows, device=device,
            sample_documents_per_language=args.sample_documents_per_language,
            store_index=args.store_index)
    elif args.task == "reranking":
        corpus = load_jsonl(args.corpus) if args.corpus is not None else None
        result = evaluate_reranking(model, rows, device=device,
                                    sample_queries=args.sample_queries,
                                    require_20_candidates=args.require_20_candidates,
                                    corpus=corpus)
    else:
        if args.length is None:
            raise ValueError("performance task requires --length")
        result = {"performance": _evaluate_performance(
            model, args.data, device=device, length=args.length,
            warmup_steps=args.warmup_steps, measured_steps=args.measured_steps)}
    result["task"] = args.task
    result["variant"] = args.variant
    result["elapsed_seconds"] = time.perf_counter() - started
    result["data_file"] = str(args.data.resolve())
    result["device"] = str(device)
    result["evaluation_scope"] = "calibration_only" if args.sample_documents_per_language else "prepared_data_scope"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.predictions_output:
        predictions = result.get("predictions", [])
        if not predictions and args.task == "retrieval":
            predictions = [prediction for language in result.get("languages", [])
                           for prediction in language.get("query_predictions", [])]
        args.predictions_output.parent.mkdir(parents=True, exist_ok=True)
        with args.predictions_output.open("w", encoding="utf-8") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "evaluation_completed", "task": args.task,
                      "variant": args.variant, "elapsed_seconds": result["elapsed_seconds"],
                      "output": str(args.output.resolve())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
