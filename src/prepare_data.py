#!/usr/bin/env python
"""Acquire pinned public sources and build tokenized pilot/full data interfaces."""

from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import re
import shutil
import time
import zlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Iterator

# HF datasets can read large parquet ranges over slower proxies; allow the read to finish.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")

import numpy as np
import requests
from huggingface_hub import HfApi, hf_hub_url
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_DIR = ROOT / "artifacts" / "mmbert-base"
PREPARED_ROOT = ROOT / "data" / "prepared"
RAW_ROOT = ROOT / "artifacts" / "raw"
SEED = 17
DATASET_SOURCES = {
    "wikipedia": {
        "id": "wikimedia/wikipedia",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "license": "CC-BY-SA-3.0 and GFDL (Hub dataset card)",
        "configs": {"en": "20231101.en", "zh": "20231101.zh"},
        "license_uri": "https://huggingface.co/datasets/wikimedia/wikipedia/tree/b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
    },
    "xnli": {
        "id": "facebook/xnli",
        "revision": "b8dd5d7af51114dbda02c0e3f6133f332186418e",
        "license": "not declared in the pinned Hub dataset metadata",
        "configs": {"en": "en", "zh": "zh"},
        "license_uri": "https://huggingface.co/datasets/facebook/xnli/tree/b8dd5d7af51114dbda02c0e3f6133f332186418e",
    },
    "squad": {
        "id": "rajpurkar/squad",
        "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
        "license": "CC-BY-SA-4.0 (Hub dataset card)",
        "configs": {"en": "plain_text"},
        "license_uri": "https://huggingface.co/datasets/rajpurkar/squad/tree/7b6d24c440a36b6815f21b70d25016731768db1f",
    },
    "cmrc2018": {
        "id": "hfl/cmrc2018",
        "revision": "137f2c45a24275fb68f6961c4d357f46288886aa",
        "license": "CC-BY-SA-4.0 (Hub dataset card)",
        "configs": {"zh": "default"},
        "license_uri": "https://huggingface.co/datasets/hfl/cmrc2018/tree/137f2c45a24275fb68f6961c4d357f46288886aa",
    },
    "mldr": {
        "id": "Shitao/MLDR",
        "revision": "d67138e705d963e346253a80e59676ddb418810a",
        "license": "MIT (Hub dataset card)",
        "configs": {"en": "mldr-v1.0-en", "zh": "mldr-v1.0-zh"},
        "license_uri": "https://huggingface.co/datasets/Shitao/MLDR/tree/d67138e705d963e346253a80e59676ddb418810a",
    },
}
WIKI_LENGTHS = (512, 1024, 4096, 8192)
FULL_WIKI_TRAIN_ROWS = {512: 8192, 1024: 4096, 4096: 512, 8192: 128}
MLDR_PREFIX_RECORDS: dict[tuple[str, str, str], dict[str, Any]] = {}
MLDR_SOURCE_METADATA: dict[str, Any] = {}


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(_portable_paths(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _portable_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_paths(item) for item in value]
    if isinstance(value, str) and value.startswith(("artifacts\\", "data\\", "src\\", "configs\\")):
        return value.replace("\\", "/")
    return value


def dump_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with temp.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    temp.replace(path)
    return count


def iter_jsonl_file(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def get_stream(dataset_id: str, config: str | None, split: str, revision: str, shuffle: bool = False):
    from datasets import load_dataset
    kwargs: dict[str, Any] = {"path": dataset_id, "split": split, "streaming": True, "revision": revision}
    if config is not None:
        kwargs["name"] = config
    stream = load_dataset(**kwargs)
    if shuffle:
        stream = stream.shuffle(seed=SEED, buffer_size=4096)
    return stream


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def wiki_split(source_id: str) -> str:
    bucket = stable_int(f"wiki-split-v1:{source_id}") % 100
    return "train" if bucket < 90 else "dev" if bucket < 95 else "test"


def qa_split(language: str, article_id: str) -> str:
    return "dev" if stable_int(f"qa-article-split-v1:{language}:{article_id}") % 10 == 0 else "train"


def text_sha(text: str) -> str:
    return sha256_bytes(" ".join(text.split()).encode("utf-8"))


def profile_targets(
    profile: str,
    qa_train: int | None,
    qa_dev: int | None,
    retrieval_train: int | None,
    classification_train: int | None = None,
    plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if profile == "pilot":
        wiki_train_per_language = 16
        wiki_eval_per_language = 8
        classification_train = classification_train or 64
        classification_eval = 64
        qa_train = min(qa_train or 64, 64)
        qa_dev = min(qa_dev or 32, 32)
        qa_eval = 32
        retrieval_train = min(retrieval_train or 64, 64)
        retrieval_dev = 16
        retrieval_test = 16
        retrieval_corpus = 64
        wiki_train_rows_by_length = {length: 32 for length in WIKI_LENGTHS}
    elif profile == "full_67m":
        plan = plan or {}
        stages = plan.get("adaptation", {}).get("stages", [])
        wiki_train_rows_by_length = {
            int(stage["max_length"]): int(stage["input_tokens"]) // int(stage["max_length"])
            for stage in stages
        }
        expected_lengths = set(WIKI_LENGTHS)
        if set(wiki_train_rows_by_length) != expected_lengths:
            raise ValueError(f"Plan stages must cover Wiki lengths {sorted(expected_lengths)}")
        if any(rows % 2 for rows in wiki_train_rows_by_length.values()):
            raise ValueError("Bilingual Wiki window targets must divide evenly between languages")
        downstream = plan.get("downstream", {})
        classification_train = classification_train or int(downstream["classification"]["train_per_language"])
        qa_train = qa_train or int(downstream["qa"]["train_questions_per_language"])
        retrieval_train = retrieval_train or int(downstream["retrieval"]["train_queries_per_language"])
        qa_dev = qa_dev or 200
        wiki_train_per_language = {length: rows // 2 for length, rows in wiki_train_rows_by_length.items()}
        wiki_eval_per_language = 32
        classification_eval = None
        qa_eval = None
        retrieval_dev = int(plan.get("evaluation", {}).get("retrieval", {}).get("dev_queries_per_language", 100))
        retrieval_test = int(plan.get("evaluation", {}).get("retrieval", {}).get("test_queries_per_language", 200))
        retrieval_corpus = int(plan.get("evaluation", {}).get("retrieval", {}).get("corpus_documents_per_language", 5000))
    else:
        classification_train = classification_train or 2000
        classification_eval = None
        qa_train = qa_train or 2000
        qa_dev = qa_dev or 200
        qa_eval = None
        retrieval_train = retrieval_train or 1000
        retrieval_dev = 100
        retrieval_test = 200
        retrieval_corpus = 5000
        wiki_train_per_language = None
        wiki_train_rows_by_length = dict(FULL_WIKI_TRAIN_ROWS)
        wiki_eval_per_language = 16
    return {
        "wiki_train_per_language": wiki_train_per_language,
        "wiki_train_rows_by_length": wiki_train_rows_by_length,
        "wiki_eval_per_language": wiki_eval_per_language,
        "classification_train": classification_train,
        "classification_eval": classification_eval,
        "qa_train": qa_train,
        "qa_dev": qa_dev,
        "qa_eval": qa_eval,
        "retrieval_train": retrieval_train,
        "retrieval_dev": retrieval_dev,
        "retrieval_test": retrieval_test,
        "retrieval_corpus": retrieval_corpus,
    }


def artifact_root(profile: str) -> Path:
    return PREPARED_ROOT if profile == "pilot" else PREPARED_ROOT / profile


def raw_root(profile: str) -> Path:
    return RAW_ROOT / profile


def save_plan_snapshot(profile: str, plan_path: Path, plan_bytes: bytes) -> tuple[str, str]:
    """Persist the exact configuration bytes used by this preparation run."""
    digest = sha256_bytes(plan_bytes)
    snapshot = raw_root(profile) / "provenance" / f"experiment_plan_{digest}.json"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.is_file():
        if sha256_file(snapshot) != digest:
            raise RuntimeError(f"Plan snapshot path exists with different content: {snapshot}")
    else:
        temporary = snapshot.with_name(f"{snapshot.name}.{os.getpid()}.tmp")
        temporary.write_bytes(plan_bytes)
        temporary.replace(snapshot)
    if sha256_file(snapshot) != digest:
        raise RuntimeError(f"Plan snapshot hash mismatch after write: {snapshot}")
    try:
        snapshot_relative = str(snapshot.relative_to(ROOT))
    except ValueError:
        snapshot_relative = str(snapshot)
    try:
        plan_relative = str(plan_path.relative_to(ROOT))
    except ValueError:
        plan_relative = str(plan_path)
    return plan_relative, snapshot_relative


def manifest_template(profile: str, tokenizer_path: Path) -> dict[str, Any]:
    tokenizer_file = tokenizer_path / "tokenizer.json"
    return {
        "manifest_version": 1,
        "profile": profile,
        "created_utc": now_utc(),
        "seed": SEED,
        "tokenizer": {
            "model_id": "jhu-clsp/mmBERT-base",
            "revision": "8c1f42296af5de623b0ad6d3a5ed6a5a1838d82c",
            "tokenizer_json_sha256": sha256_file(tokenizer_file) if tokenizer_file.is_file() else None,
            "path": str(tokenizer_file.relative_to(ROOT)) if tokenizer_file.is_file() else None,
        },
        "sources": [
            {
                "key": key,
                "dataset_id": value["id"],
                "revision": value["revision"],
                "configs": value["configs"],
                "license_status": value["license"],
                "license_metadata_uri": value["license_uri"],
            }
            for key, value in DATASET_SOURCES.items()
        ],
        "artifacts": {},
        "notes": [
            "Only source records actually read and saved are counted; no full upstream repository snapshot is implied.",
            "Wiki windows are non-overlapping, full-length, article-local, with BOS/EOS and no padding.",
            "Wiki split is deterministic by article ID; all windows from an article remain in one split.",
            "Official XNLI validation/test and public SQuAD/CMRC validation are distinguished from training-derived development rows.",
            "MLDR gzip JSONL is read directly at the pinned revision because the repository's dataset script is not supported by the installed datasets version.",
        ],
    }


def save_manifest(profile: str, manifest: dict[str, Any], filename: str = "manifest.json") -> Path:
    path = artifact_root(profile) / filename
    manifest["updated_utc"] = now_utc()
    dump_json(path, manifest)
    return path


def merge_task_manifests(profile: str) -> Path:
    out_root = artifact_root(profile)
    combined = manifest_template(profile, TOKENIZER_DIR)
    combined["artifacts"] = {}
    task_files = []
    for task in ("wiki", "classification", "qa", "retrieval"):
        path = out_root / f"manifest_{task}.json"
        if not path.is_file():
            continue
        part = json.loads(path.read_text(encoding="utf-8"))
        combined["artifacts"].update(part.get("artifacts", {}))
        if "data_profile" in part:
            combined["data_profile"] = part["data_profile"]
        if "qa_answer_offset_alignment" in part:
            combined["qa_answer_offset_alignment"] = part["qa_answer_offset_alignment"]
        if "experiment_plan" in part:
            combined["experiment_plan"] = part["experiment_plan"]
        task_files.append(path.name)
    combined["task_manifests"] = task_files
    return save_manifest(profile, combined, "manifest.json")


def reuse_manifest_artifact(
    source_profile: str,
    target_profile: str,
    task: str,
    artifact_key: str,
    manifest: dict[str, Any],
    target_key: str | None = None,
) -> dict[str, Any]:
    """Hash-check and link a fixed artifact from an immutable earlier profile."""
    source_manifest_path = artifact_root(source_profile) / f"manifest_{task}.json"
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    item = dict(source_manifest["artifacts"][artifact_key])
    for path_key, hash_key in (("path", "sha256"), ("raw_path", "raw_sha256")):
        old_relative = item.get(path_key)
        if not old_relative:
            continue
        source_path = ROOT / old_relative
        expected_hash = item.get(hash_key)
        if not source_path.is_file() or not expected_hash:
            raise FileNotFoundError(f"Cannot reuse {artifact_key}: missing source or recorded hash at {old_relative}")
        actual_hash = sha256_file(source_path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Refusing to reuse {old_relative}: SHA-256 {actual_hash} != {expected_hash}")
        parts = Path(old_relative).parts
        if parts[:2] == ("data", "prepared"):
            destination = artifact_root(target_profile) / Path(*parts[3:])
        elif parts[:2] == ("artifacts", "raw"):
            destination = raw_root(target_profile) / Path(*parts[3:])
        else:
            raise ValueError(f"Unsupported reused artifact path: {old_relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".reuse.tmp")
        if temporary.exists():
            temporary.unlink()
        try:
            os.link(source_path, temporary)
        except OSError:
            shutil.copy2(source_path, temporary)
        if sha256_file(temporary) != expected_hash:
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"Reused copy failed SHA-256 verification: {old_relative}")
        temporary.replace(destination)
        item[path_key] = str(destination.relative_to(ROOT))
    item["reused_from_profile"] = source_profile
    manifest["artifacts"][target_key or artifact_key] = item
    return item


def split_existing_wiki_files(profile: str) -> Path:
    """Derive train-only primary arrays and held-out arrays from an earlier combined cache."""
    out_root = artifact_root(profile)
    manifest_path = out_root / "manifest_wiki.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else manifest_template(profile, TOKENIZER_DIR)
    for length in WIKI_LENGTHS:
        npy_path = out_root / f"wiki_{length}.npy"
        metadata_path = out_root / f"wiki_{length}.jsonl"
        array = np.load(npy_path, allow_pickle=False)
        rows = [json.loads(line) for line in metadata_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(array) != len(rows):
            raise RuntimeError(f"Wiki array/metadata row mismatch for length {length}: {len(array)} != {len(rows)}")
        split_counts: dict[str, dict[str, int]] = {}
        split_records = {}
        for split in ("train", "dev", "test"):
            indices = [i for i, row in enumerate(rows) if row["split"] == split]
            split_array = array[indices] if indices else np.empty((0, length), dtype=np.int32)
            split_rows = [rows[i] for i in indices]
            suffix = "" if split == "train" else f"_{split}"
            split_npy = out_root / f"wiki_{length}{suffix}.npy"
            tmp_npy = split_npy.with_suffix(".npy.tmp")
            with tmp_npy.open("wb") as stream:
                np.save(stream, split_array, allow_pickle=False)
            tmp_npy.replace(split_npy)
            split_jsonl = out_root / f"wiki_{length}{suffix}.jsonl"
            dump_jsonl(split_jsonl, split_rows)
            split_counts[split] = {lang: sum(1 for row in split_rows if row["language"] == lang) for lang in ("en", "zh")}
            split_records[split] = {
                "status": "complete" if len(split_rows) else "empty",
                "npy_path": str(split_npy.relative_to(ROOT)),
                "npy_sha256": sha256_file(split_npy),
                "metadata_path": str(split_jsonl.relative_to(ROOT)),
                "metadata_sha256": sha256_file(split_jsonl),
                "shape": list(split_array.shape),
                "dtype": "int32",
                "language_counts": split_counts[split],
                "tokens_per_row": length,
                "rows": len(split_rows),
                "padding_rows": 0,
            }
        entry = manifest.setdefault("artifacts", {}).setdefault(f"wiki_{length}", {})
        entry.update(
            {
                "status": "complete" if split_counts["train"]["en"] + split_counts["train"]["zh"] >= 32 else "partial",
                "npy_path": split_records["train"]["npy_path"],
                "npy_sha256": split_records["train"]["npy_sha256"],
                "metadata_path": split_records["train"]["metadata_path"],
                "metadata_sha256": split_records["train"]["metadata_sha256"],
                "shape": split_records["train"]["shape"],
                "dtype": "int32",
                "split_language_counts": split_counts,
                "tokens_per_row": length,
                "full_window_rows": split_records["train"]["rows"],
                "padding_rows": 0,
                "evaluation_splits": {key: value for key, value in split_records.items() if key != "train"},
            }
        )
    manifest.setdefault("artifacts", {}).setdefault("wiki_profile", {})["train_arrays_exclude_holdouts"] = True
    # Earlier pilot files were written before raw assets were partitioned by profile.
    if profile != "full_67m":
        for task in ("wikipedia", "xnli", "qa", "mldr"):
            old_dir = RAW_ROOT / task
            new_dir = raw_root(profile) / task
            if old_dir.is_dir() and not new_dir.exists():
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                old_dir.replace(new_dir)
            elif old_dir.is_dir() and new_dir.is_dir():
                for old_file in old_dir.iterdir():
                    new_file = new_dir / old_file.name
                    if old_file.is_file() and not new_file.exists():
                        old_file.replace(new_file)
    for artifact in manifest.get("artifacts", {}).values():
        if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
            artifact["path"] = artifact["path"].replace("artifacts/raw/", f"artifacts/raw/{profile}/")
        if isinstance(artifact, dict) and isinstance(artifact.get("raw_path"), str):
            artifact["raw_path"] = artifact["raw_path"].replace("artifacts/raw/", f"artifacts/raw/{profile}/")
    for other_manifest in out_root.glob("manifest_*.json"):
        if other_manifest.name == "manifest_wiki.json":
            continue
        other_data = json.loads(other_manifest.read_text(encoding="utf-8"))
        for artifact in other_data.get("artifacts", {}).values():
            if isinstance(artifact, dict) and isinstance(artifact.get("path"), str):
                artifact["path"] = artifact["path"].replace("artifacts/raw/", f"artifacts/raw/{profile}/")
            if isinstance(artifact, dict) and isinstance(artifact.get("raw_path"), str):
                artifact["raw_path"] = artifact["raw_path"].replace("artifacts/raw/", f"artifacts/raw/{profile}/")
        dump_json(other_manifest, other_data)
    return save_manifest(profile, manifest, "manifest_wiki.json")


def prepare_wiki(tokenizer, profile: str, manifest: dict[str, Any], data_targets: dict[str, Any]) -> None:
    out_root = artifact_root(profile)
    out_root.mkdir(parents=True, exist_ok=True)
    if profile == "pilot":
        train_per_language = {length: 16 for length in WIKI_LENGTHS}
        eval_per_language = {"dev": 8, "test": 8}
    elif profile == "full_67m":
        train_per_language = data_targets["wiki_train_per_language"]
        eval_per_language = {"dev": 32, "test": 32}
    else:
        train_per_language = {length: FULL_WIKI_TRAIN_ROWS[length] // 2 for length in WIKI_LENGTHS}
        eval_per_language = {"dev": 16, "test": 16}
    split_targets: dict[int, dict[str, dict[str, int]]] = {}
    for length in WIKI_LENGTHS:
        split_targets[length] = {
            "train": {"en": train_per_language[length], "zh": train_per_language[length]},
            "dev": {"en": eval_per_language["dev"], "zh": eval_per_language["dev"]},
            "test": {"en": eval_per_language["test"], "zh": eval_per_language["test"]},
        }
    sequences: dict[int, list[list[int]]] = {length: [] for length in WIKI_LENGTHS}
    metadata: dict[int, list[dict[str, Any]]] = {length: [] for length in WIKI_LENGTHS}
    counts: dict[int, Counter[str]] = {length: Counter() for length in WIKI_LENGTHS}
    raw_docs: dict[str, list[dict[str, Any]]] = {"en": [], "zh": []}
    seen_hashes: set[str] = set()
    exhausted: set[str] = set()
    streams = {
        lang: iter(
            get_stream(
                DATASET_SOURCES["wikipedia"]["id"],
                DATASET_SOURCES["wikipedia"]["configs"][lang],
                "train",
                DATASET_SOURCES["wikipedia"]["revision"],
                shuffle=(profile == "pilot"),
            )
        )
        for lang in ("en", "zh")
    }
    processed = Counter()
    t0 = time.monotonic()
    print(f"[{profile}] Preparing bilingual Wikipedia full windows.", flush=True)
    while True:
        missing = [
            (length, split, lang)
            for length in WIKI_LENGTHS
            for split in ("train", "dev", "test")
            for lang in ("en", "zh")
            if counts[length][f"{split}:{lang}"] < split_targets[length][split][lang]
        ]
        if not missing:
            break
        progressed = False
        for lang in ("en", "zh"):
            if lang in exhausted or not any(item[2] == lang for item in missing):
                continue
            try:
                article = next(streams[lang])
            except StopIteration:
                exhausted.add(lang)
                continue
            processed[lang] += 1
            text = str(article.get("text") or "").strip()
            if not text:
                continue
            source_id = f"{lang}:{article.get('id') or article.get('url') or text_sha(text)}"
            raw_digest = text_sha(text)
            dedup_key = f"{lang}:{raw_digest}"
            if dedup_key in seen_hashes:
                continue
            seen_hashes.add(dedup_key)
            split = wiki_split(source_id)
            needed_for_article = {
                length: split_targets[length][split][lang] - counts[length][f"{split}:{lang}"]
                for length in WIKI_LENGTHS
            }
            if not any(value > 0 for value in needed_for_article.values()):
                continue
            token_ids = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max(WIKI_LENGTHS) * 4,
            )["input_ids"]
            accepted = False
            for length in WIKI_LENGTHS:
                remaining = needed_for_article[length]
                if remaining <= 0:
                    continue
                width = length - 2
                article_windows = 0
                for offset in range(0, len(token_ids) - width + 1, width):
                    if article_windows >= 8 or counts[length][f"{split}:{lang}"] >= split_targets[length][split][lang]:
                        break
                    content = token_ids[offset : offset + width]
                    if len(content) != width:
                        continue
                    sequences[length].append([tokenizer.bos_token_id, *content, tokenizer.eos_token_id])
                    metadata[length].append(
                        {
                            "id": f"{source_id}:w{length}:{offset // width}",
                            "source_id": source_id,
                            "language": lang,
                            "split": split,
                            "source_revision": DATASET_SOURCES["wikipedia"]["revision"],
                            "chunk_index": offset // width,
                            "content_tokens": width,
                            "window_length": length,
                            "title": article.get("title"),
                            "url": article.get("url"),
                            "raw_text_sha256": raw_digest,
                        }
                    )
                    counts[length][f"{split}:{lang}"] += 1
                    article_windows += 1
                    accepted = True
            if accepted:
                raw_docs[lang].append(
                    {
                        "source_id": source_id,
                        "language": lang,
                        "split": split,
                        "revision": DATASET_SOURCES["wikipedia"]["revision"],
                        "id": article.get("id"),
                        "title": article.get("title"),
                        "url": article.get("url"),
                        "text_sha256": raw_digest,
                        "text": text,
                    }
                )
                progressed = True
            if processed[lang] % 1000 == 0:
                print(f"[{profile}] Wikipedia articles read {dict(processed)}; train 512 rows {counts[512]['train:en'] + counts[512]['train:zh']}/{sum(split_targets[512]['train'].values())}.", flush=True)
        if not progressed and exhausted == {"en", "zh"}:
            break
        if exhausted == {"en", "zh"}:
            break

    status = "complete"
    for length in WIKI_LENGTHS:
        all_array = np.asarray(sequences[length], dtype=np.int32)
        if all_array.size:
            all_array = all_array.reshape((-1, length))
        else:
            all_array = np.empty((0, length), dtype=np.int32)
        length_counts = {
            split: {
                lang: counts[length][f"{split}:{lang}"]
                for lang in ("en", "zh")
            }
            for split in ("train", "dev", "test")
        }
        expected_train = sum(split_targets[length]["train"].values())
        actual_train = length_counts["train"]["en"] + length_counts["train"]["zh"]
        if actual_train < min(32, expected_train) or any(
            length_counts[split][lang] < split_targets[length][split][lang]
            for split in ("train", "dev", "test")
            for lang in ("en", "zh")
        ):
            status = "incomplete"
        train_rows = []
        split_artifacts = {}
        for split in ("train", "dev", "test"):
            selected_indices = [i for i, item in enumerate(metadata[length]) if item["split"] == split]
            split_array = all_array[selected_indices] if selected_indices else np.empty((0, length), dtype=np.int32)
            split_metadata = [metadata[length][i] for i in selected_indices]
            suffix = "" if split == "train" else f"_{split}"
            npy_path = out_root / f"wiki_{length}{suffix}.npy"
            temp_npy = npy_path.with_suffix(".npy.tmp")
            with temp_npy.open("wb") as stream:
                np.save(stream, split_array, allow_pickle=False)
            temp_npy.replace(npy_path)
            meta_path = out_root / f"wiki_{length}{suffix}.jsonl"
            dump_jsonl(meta_path, split_metadata)
            if split == "train":
                train_rows = split_metadata
            split_artifacts[split] = {
                "status": "complete" if all(length_counts[split][lang] == split_targets[length][split][lang] for lang in ("en", "zh")) else "partial",
                "npy_path": str(npy_path.relative_to(ROOT)),
                "npy_sha256": sha256_file(npy_path),
                "metadata_path": str(meta_path.relative_to(ROOT)),
                "metadata_sha256": sha256_file(meta_path),
                "shape": list(split_array.shape),
                "dtype": "int32",
                "language_counts": length_counts[split],
                "tokens_per_row": length,
                "rows": int(split_array.shape[0]),
                "padding_rows": 0,
                "targets": split_targets[length][split],
            }
        manifest["artifacts"][f"wiki_{length}"] = {
            "status": "complete" if all(length_counts[s][l] == split_targets[length][s][l] for s in split_targets[length] for l in split_targets[length][s]) else "partial",
            "npy_path": split_artifacts["train"]["npy_path"],
            "npy_sha256": split_artifacts["train"]["npy_sha256"],
            "metadata_path": split_artifacts["train"]["metadata_path"],
            "metadata_sha256": split_artifacts["train"]["metadata_sha256"],
            "shape": split_artifacts["train"]["shape"],
            "dtype": "int32",
            "split_language_counts": length_counts,
            "tokens_per_row": length,
            "full_window_rows": split_artifacts["train"]["rows"],
            "padding_rows": 0,
            "targets": split_targets[length]["train"],
            "evaluation_splits": {
                split: {key: value for key, value in artifact.items() if key != "targets"}
                for split, artifact in split_artifacts.items()
                if split != "train"
            },
        }
    for lang in ("en", "zh"):
        raw_path = raw_root(profile) / "wikipedia" / f"selected_{lang}.jsonl"
        dump_jsonl(raw_path, raw_docs[lang])
        manifest["artifacts"][f"wiki_source_{lang}"] = {
            "path": str(raw_path.relative_to(ROOT)),
            "rows": len(raw_docs[lang]),
            "sha256": sha256_file(raw_path),
            "revision": DATASET_SOURCES["wikipedia"]["revision"],
            "config": DATASET_SOURCES["wikipedia"]["configs"][lang],
            "article_level_split_policy": "sha256(source_id) modulo 100: train 0-89, dev 90-94, test 95-99",
            "stream_shuffle_seed": SEED,
            "stream_shuffle_buffer": 4096 if profile == "pilot" else 0,
            "source_order": "deterministic buffered shuffle" if profile == "pilot" else "pinned shard order",
            "source_articles_examined": processed[lang],
        }
    manifest["artifacts"]["wiki_profile"] = {
        "status": status,
        "article_count_by_language": {lang: len(raw_docs[lang]) for lang in ("en", "zh")},
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "source_ids_shared_across_variants": True,
    }
    print(f"[{profile}] Wiki rows: " + json.dumps({k: v["full_window_rows"] for k, v in manifest["artifacts"].items() if k.startswith("wiki_") and "full_window_rows" in v}), flush=True)


def prepare_classification(tokenizer, profile: str, targets: dict[str, Any], manifest: dict[str, Any]) -> None:
    out_root = artifact_root(profile)
    out_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "dev", "test")}
    raw_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    if profile == "full_67m":
        for split in ("dev", "test"):
            reuse_manifest_artifact("full", profile, "classification", f"classification_{split}", manifest)
        for lang, source_split in (("en", "validation"), ("zh", "validation"), ("en", "test"), ("zh", "test")):
            reuse_manifest_artifact("full", profile, "classification", f"raw_xnli_{lang}_{source_split}", manifest)
    for lang in ("en", "zh"):
        source = DATASET_SOURCES["xnli"]
        for output_split, source_split in (("train", "train"), ("dev", "validation"), ("test", "test")):
            if profile == "full_67m" and output_split != "train":
                continue
            limit = targets["classification_train"] if output_split == "train" else targets["classification_eval"]
            stream = get_stream(source["id"], source["configs"][lang], source_split, source["revision"], shuffle=output_split == "train")
            count = 0
            for index, row in enumerate(stream):
                if limit is not None and count >= limit:
                    break
                premise = str(row.get("premise") or "")
                hypothesis = str(row.get("hypothesis") or "")
                label = row.get("label")
                if not premise or not hypothesis or label is None or int(label) < 0:
                    continue
                encoded = tokenizer(premise, hypothesis, truncation=True, max_length=256)
                sample_id = str(row.get("idx") or f"{lang}:{source_split}:{index}")
                output = {
                    "input_ids": encoded["input_ids"],
                    "label": int(label),
                    "language": lang,
                    "id": f"{lang}:{sample_id}",
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "source_id": source["id"],
                    "source_revision": source["revision"],
                    "source_split": source_split,
                }
                out_rows[output_split].append(output)
                raw_rows[f"{lang}_{source_split}"].append(
                    {
                        "id": output["id"],
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": int(label),
                        "language": lang,
                        "source_revision": source["revision"],
                        "source_split": source_split,
                    }
                )
                count += 1
            counts[f"{output_split}:{lang}"] = count
    for split, rows in out_rows.items():
        if profile == "full_67m" and split != "train":
            continue
        path = out_root / f"classification_{split}.jsonl"
        dump_jsonl(path, rows)
        manifest["artifacts"][f"classification_{split}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "language_counts": {lang: counts[f"{split}:{lang}"] for lang in ("en", "zh")},
            "max_length": 256,
        }
    for key, rows in raw_rows.items():
        raw_path = raw_root(profile) / "xnli" / f"{key}.jsonl"
        dump_jsonl(raw_path, rows)
        manifest["artifacts"][f"raw_xnli_{key}"] = {
            "path": str(raw_path.relative_to(ROOT)),
            "rows": len(rows),
            "sha256": sha256_file(raw_path),
            "revision": DATASET_SOURCES["xnli"]["revision"],
        }


def answer_pair(row: dict[str, Any]) -> tuple[str, int] | None:
    answers = row.get("answers") or {}
    texts = answers.get("text") or []
    starts = answers.get("answer_start") or []
    for text, start in zip(texts, starts):
        text = str(text)
        if text and start is not None:
            return text, int(start)
    return None


def normalize_qa_answers(row: dict[str, Any], preserve_unalignable: bool = False) -> tuple[dict[str, Any], str, str | None, int, int]:
    """Correct answer offsets only when a unique nearest exact text match exists.

    Returns a copied row, status, reason, number of non-empty answer pairs, and
    number of offsets corrected. Unalignable evaluation rows retain their source
    answers and are marked so scoring can keep them in the denominator.
    """
    normalized = dict(row)
    context = str(row.get("context") or "")
    answers = row.get("answers") or {}
    original_answers = copy.deepcopy(row.get("source_answers_original", answers))
    texts = [str(value) for value in (answers.get("text") or [])]
    starts = list(answers.get("answer_start") or [])
    nonempty_count = sum(bool(text) for text in texts)
    normalized["source_answers_original"] = original_answers

    if nonempty_count == 0:
        normalized["answers"] = copy.deepcopy(answers)
        normalized["answer_alignment_status"] = "no_gold_answer"
        normalized["answer_alignment_reason"] = "no_nonempty_answer_text"
        return normalized, "no_gold_answer", "no_nonempty_answer_text", 0, 0

    if len(texts) != len(starts):
        reason = "answer_text_start_count_mismatch"
        normalized["answers"] = original_answers
        normalized["answer_alignment_status"] = "unalignable_eval" if preserve_unalignable else "unalignable_train"
        normalized["answer_alignment_reason"] = reason
        return normalized, normalized["answer_alignment_status"], reason, nonempty_count, 0

    aligned_starts: list[int] = []
    corrected_count = 0
    reason: str | None = None
    for text, raw_start in zip(texts, starts):
        try:
            start = int(raw_start)
        except (TypeError, ValueError):
            reason = "invalid_answer_start"
            break
        if not text:
            aligned_starts.append(start)
            continue
        end = start + len(text)
        if text and 0 <= start <= len(context) and context[start:end] == text:
            aligned_starts.append(start)
            continue

        positions: list[int] = []
        cursor = 0
        while text:
            position = context.find(text, cursor)
            if position < 0:
                break
            positions.append(position)
            cursor = position + 1
        if not positions:
            reason = "no_exact_answer_occurrence"
            break
        distance = min(abs(position - start) for position in positions)
        nearest = [position for position in positions if abs(position - start) == distance]
        if len(nearest) != 1:
            reason = "ambiguous_nearest_answer_occurrence"
            break
        aligned_starts.append(nearest[0])
        corrected_count += 1

    if reason is not None:
        normalized["answers"] = original_answers
        normalized["answer_alignment_status"] = "unalignable_eval" if preserve_unalignable else "unalignable_train"
        normalized["answer_alignment_reason"] = reason
        return normalized, normalized["answer_alignment_status"], reason, nonempty_count, 0

    normalized_answers = copy.deepcopy(answers)
    normalized_answers["text"] = texts
    normalized_answers["answer_start"] = aligned_starts
    normalized["answers"] = normalized_answers
    source_starts = list(original_answers.get("answer_start") or [])
    if len(source_starts) == len(aligned_starts):
        corrected_count = sum(1 for original, aligned in zip(source_starts, aligned_starts) if str(original) != str(aligned))
    status = "corrected_nearest" if corrected_count else "aligned"
    normalized["answer_alignment_status"] = status
    normalized["answer_alignment_reason"] = "nearest_exact_answer_occurrence" if corrected_count else "exact_source_offset"
    return normalized, status, normalized["answer_alignment_reason"], nonempty_count, corrected_count


def encode_qa_example(
    tokenizer,
    row: dict[str, Any],
    lang: str,
    source_id: str,
    source_revision: str,
    source_split: str,
    training: bool = False,
) -> list[dict[str, Any]]:
    pair = answer_pair(row)
    context = str(row.get("context") or "")
    question = str(row.get("question") or "")
    if (pair is None and training) or (not context and training) or (not question and training):
        return []
    if pair is None:
        answer_text, answer_start_char, answer_end_char = "", -1, -1
    else:
        answer_text, answer_start_char = pair
        answer_end_char = answer_start_char + len(answer_text)
    alignment_status = str(row.get("answer_alignment_status") or "aligned")
    alignment_reason = row.get("answer_alignment_reason")
    answer_is_aligned = alignment_status in {"aligned", "corrected_nearest"}
    if answer_is_aligned and not (0 <= answer_start_char < answer_end_char <= len(context)):
        return []
    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding=False,
    )
    features = []
    for feature_index, input_ids in enumerate(encoded["input_ids"]):
        offsets = encoded["offset_mapping"][feature_index]
        sequence_ids = encoded.sequence_ids(feature_index)
        context_positions = [i for i, segment in enumerate(sequence_ids) if segment == 1]
        start_candidates = [
            i for i in context_positions
            if offsets[i][0] <= answer_start_char < offsets[i][1]
        ]
        end_candidates = [
            i for i in context_positions
            if offsets[i][0] < answer_end_char <= offsets[i][1]
        ]
        if answer_is_aligned and start_candidates and end_candidates:
            start_position = start_candidates[0]
            end_position = end_candidates[-1]
            if start_position > end_position:
                start_position = end_position = -100
        else:
            start_position = end_position = -100
        all_answers = row.get("answers") or {}
        sample_id = str(row.get("id") or f"{lang}:{source_split}:{stable_int(question)}")
        feature = {
            "input_ids": input_ids,
            "start_positions": start_position,
            "end_positions": end_position,
            "context_mask": [1 if segment == 1 else 0 for segment in sequence_ids],
            "offset_mapping": [[int(a), int(b)] for a, b in offsets],
            "language": lang,
            "id": sample_id,
            "question_id": sample_id,
            "feature_id": f"{sample_id}:window:{feature_index}",
            "context": context,
            "question": question,
            "answer_text": answer_text,
            "answer_start_char": answer_start_char,
            "answer_end_char": answer_end_char,
            "answers": all_answers,
            "source_answers_original": row.get("source_answers_original", all_answers),
            "answer_alignment_status": alignment_status,
            "answer_alignment_reason": alignment_reason,
            "title": row.get("title"),
            "source_id": source_id,
            "source_revision": source_revision,
            "source_split": source_split,
            "feature_index": feature_index,
        }
        features.append(feature)
    if training:
        return [feature for feature in features if feature["start_positions"] >= 0][:1]
    # Preserve every overflow window for evaluation. A -100 label means the answer
    # span is outside that window; all such windows remain in the question denominator.
    return features


def make_long_qa_example(tokenizer, base: dict[str, Any], background_docs: list[dict[str, Any]], position: str) -> dict[str, Any] | None:
    question = base["question"]
    original_context = base["context"]
    answer_text = base["answer_text"]
    original_start = int(base["answer_start_char"])
    original_end = int(base["answer_end_char"])
    if original_context[original_start:original_end] != answer_text:
        located = original_context.find(answer_text)
        if located < 0:
            return None
        original_start = located
        original_end = located + len(answer_text)

    question_tokens = len(tokenizer(question, add_special_tokens=False)["input_ids"])
    pair_specials = tokenizer.num_special_tokens_to_add(pair=True)
    context_budget = 4096 - question_tokens - pair_specials
    base_tokens = tokenizer(original_context, add_special_tokens=False)["input_ids"]
    background_budget = context_budget - len(base_tokens) - 8
    if background_budget < 256:
        return None

    start_index = stable_int(f"qa-long-bg-v1:{SEED}:{base['question_id']}") % max(1, len(background_docs))
    background_ids: list[int] = []
    source_ids: list[str] = []
    for offset in range(len(background_docs)):
        doc = background_docs[(start_index + offset) % len(background_docs)]
        doc_ids = tokenizer(
            str(doc.get("text") or ""),
            add_special_tokens=False,
            truncation=True,
            max_length=max(512, background_budget - len(background_ids) + 64),
        )["input_ids"]
        if not doc_ids:
            continue
        background_ids.extend(doc_ids)
        source_ids.append(str(doc.get("source_id") or doc.get("id") or "unknown"))
        if len(background_ids) >= background_budget:
            break
    if len(background_ids) < background_budget // 2:
        return None
    background_ids = background_ids[:background_budget]
    if position == "first":
        prefix_count = int(len(background_ids) * 0.05)
    elif position == "last":
        prefix_count = int(len(background_ids) * 0.95)
    else:
        prefix_count = len(background_ids) // 2
    prefix_text = tokenizer.decode(background_ids[:prefix_count], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    suffix_text = tokenizer.decode(background_ids[prefix_count:], skip_special_tokens=True, clean_up_tokenization_spaces=False)
    separator = "\n\n"
    context = f"{prefix_text}{separator}{original_context}{separator}{suffix_text}"
    answer_start = len(prefix_text) + len(separator) + original_start
    answer_end = answer_start + len(answer_text)

    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=4096,
        return_offsets_mapping=True,
        padding=False,
    )
    offsets = encoded["offset_mapping"]
    sequence_ids = encoded.sequence_ids()
    context_positions = [i for i, segment in enumerate(sequence_ids) if segment == 1]
    start_candidates = [i for i in context_positions if offsets[i][0] <= answer_start < offsets[i][1]]
    end_candidates = [i for i in context_positions if offsets[i][0] < answer_end <= offsets[i][1]]
    if not start_candidates or not end_candidates or len(encoded["input_ids"]) < 3500:
        return None
    sample_id = f"{base['question_id']}:long:{position}"
    answer_bundle = {"text": [answer_text], "answer_start": [answer_start]}
    return {
        "input_ids": encoded["input_ids"],
        "start_positions": start_candidates[0],
        "end_positions": end_candidates[-1],
        "context_mask": [1 if segment == 1 else 0 for segment in sequence_ids],
        "offset_mapping": [[int(a), int(b)] for a, b in offsets],
        "language": base["language"],
        "id": sample_id,
        "question_id": base["question_id"],
        "feature_id": sample_id,
        "context": context,
        "original_context": original_context,
        "question": question,
        "answer_text": answer_text,
        "answer_start_char": answer_start,
        "answer_end_char": answer_end,
        "answers": answer_bundle,
        "source_answers_original": base.get("source_answers_original", base.get("answers")),
        "answer_alignment_status": base.get("answer_alignment_status", "aligned"),
        "answer_alignment_reason": base.get("answer_alignment_reason"),
        "title": base.get("title"),
        "source_id": base["source_id"],
        "source_revision": base["source_revision"],
        "source_split": "train",
        "feature_index": 0,
        "window_type": "long",
        "evidence_position": position,
        "input_length": len(encoded["input_ids"]),
        "background_source_ids": source_ids,
        "background_source_revision": DATASET_SOURCES["wikipedia"]["revision"],
    }


def prepare_long_qa(tokenizer, profile: str, targets: dict[str, Any], train_rows: list[dict[str, Any]], manifest: dict[str, Any]) -> list[dict[str, Any]]:
    background_by_lang: dict[str, list[dict[str, Any]]] = {}
    background_profile = profile
    for lang in ("en", "zh"):
        path = raw_root(profile) / "wikipedia" / f"selected_{lang}.jsonl"
        if not path.is_file() and profile != "pilot":
            source_profile = "full" if profile == "full_67m" else "pilot"
            path = raw_root(source_profile) / "wikipedia" / f"selected_{lang}.jsonl"
            background_profile = source_profile
        if not path.is_file():
            raise RuntimeError(f"Long QA augmentation needs prepared Wiki source backgrounds: {path}")
        background_by_lang[lang] = [
            row
            for row in iter_jsonl_file(path)
            if row.get("split", "train") == "train"
        ]
    if not all(background_by_lang.values()):
        raise RuntimeError("Long QA augmentation has no retained Wiki background documents")

    quotas = {lang: (16 if profile == "pilot" else math.ceil(targets["qa_train"] * 0.1)) for lang in ("en", "zh")}
    long_rows = []
    counts: Counter[str] = Counter()
    seen_questions = {"en": set(), "zh": set()}
    for lang in ("en", "zh"):
        for base in train_rows:
            if counts[lang] >= quotas[lang]:
                break
            if base["language"] != lang:
                continue
            if base["question_id"] in seen_questions[lang]:
                continue
            seen_questions[lang].add(base["question_id"])
            position = ("first", "middle", "last")[counts[lang] % 3]
            example = make_long_qa_example(tokenizer, base, background_by_lang[lang], position)
            if example is not None:
                long_rows.append(example)
                counts[lang] += 1
    train_rows.extend(long_rows)
    out_root = artifact_root(profile)
    long_path = out_root / "qa_long_augmentation.jsonl"
    dump_jsonl(long_path, long_rows)
    manifest["artifacts"]["qa_long_augmentation"] = {
        "path": str(long_path.relative_to(ROOT)),
        "sha256": sha256_file(long_path),
        "rows": len(long_rows),
        "language_counts": {lang: counts[lang] for lang in ("en", "zh")},
        "requested_per_language": quotas,
        "max_length": 4096,
        "window_type": "long",
        "evidence_positions": {pos: sum(1 for row in long_rows if row["evidence_position"] == pos) for pos in ("first", "middle", "last")},
        "background_source": DATASET_SOURCES["wikipedia"]["id"],
        "background_revision": DATASET_SOURCES["wikipedia"]["revision"],
        "background_data_profile": background_profile,
        "background_split": "train only",
        "source_separation": "QA evidence stays from SQuAD/CMRC; filler paragraphs come from pinned Wikipedia articles",
    }
    return long_rows


def prepare_qa(tokenizer, profile: str, targets: dict[str, Any], manifest: dict[str, Any]) -> None:
    out_root = artifact_root(profile)
    output_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "dev", "test")}
    raw_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    feature_counts: Counter[str] = Counter()
    alignment_stats: dict[str, Counter[str]] = defaultdict(Counter)
    alignment_reasons: dict[str, Counter[str]] = defaultdict(Counter)
    input_sources: dict[str, str] = {}
    rejected_rows: dict[str, list[dict[str, Any]]] = {"en": [], "zh": []}
    raw_seen: set[tuple[str, str, str]] = set()
    source_for_lang = {
        "en": ("squad", DATASET_SOURCES["squad"]["configs"]["en"]),
        "zh": ("cmrc2018", DATASET_SOURCES["cmrc2018"]["configs"]["zh"]),
    }
    if profile == "full_67m":
        old_qa_manifest = json.loads((artifact_root("full") / "manifest_qa.json").read_text(encoding="utf-8"))
        for split in ("dev", "test"):
            item = reuse_manifest_artifact("full", profile, "qa", f"qa_{split}", manifest)
            for lang in ("en", "zh"):
                counts[f"{split}:{lang}"] = int(item.get("question_counts", {}).get(lang, 0))
                feature_counts[f"{split}:{lang}"] = int(item.get("feature_counts", {}).get(lang, 0))
        for lang in ("en", "zh"):
            reuse_manifest_artifact("full", profile, "qa", f"raw_qa_{lang}_train_dev", manifest)
            reuse_manifest_artifact("full", profile, "qa", f"raw_qa_{lang}_public_validation", manifest)
            input_sources[lang] = "pinned stream for expanded train; fixed dev/public evaluation reused from full"
        old_alignment = old_qa_manifest.get("qa_answer_offset_alignment", {})
        for key, values in old_alignment.get("counts_by_output_split_and_language", {}).items():
            if key.startswith(("dev:", "test:")):
                alignment_stats[key].update(values)
        for key, values in old_alignment.get("reasons_by_output_split_and_language", {}).items():
            if key.startswith(("dev:", "test:")):
                alignment_reasons[key].update(values)
    for lang, (source_key, config) in source_for_lang.items():
        source = DATASET_SOURCES[source_key]
        train_limit = targets["qa_train"]
        dev_limit = targets["qa_dev"]
        train_count = 0
        dev_count = dev_limit if profile == "full_67m" else 0
        train_cache = raw_root(profile) / "qa" / f"{lang}_train_train.jsonl"
        dev_cache = raw_root(profile) / "qa" / f"{lang}_train_dev.jsonl"
        if train_cache.is_file() and dev_cache.is_file():
            cached_train_rows = list(iter_jsonl_file(train_cache))
            cached_dev_rows = list(iter_jsonl_file(dev_cache))
            source_rows = chain(
                ((row, "train") for row in cached_train_rows),
                ((row, "dev") for row in cached_dev_rows),
            )
            input_sources[lang] = "profile-local selected QA JSONL"
        else:
            stream = get_stream(source["id"], config, "train", source["revision"], shuffle=True)

            def stream_selected_rows():
                for source_row in stream:
                    article_id = str(source_row.get("title") or source_row.get("context_id") or "").strip()
                    if not article_id:
                        raw_id = str(source_row.get("id") or "")
                        if lang == "zh" and "_QUERY_" in raw_id:
                            article_id = raw_id.split("_QUERY_", 1)[0]
                        else:
                            article_id = raw_id
                    yield source_row, qa_split(lang, article_id) if article_id else "train"

            source_rows = stream_selected_rows()
            input_sources[lang] = "pinned Hugging Face streaming source"
        for row, split in source_rows:
            if split == "train" and train_count >= train_limit:
                continue
            if split == "dev" and dev_count >= dev_limit:
                continue
            normalized_row, alignment_status, alignment_reason, answer_pair_count, corrected_pair_count = normalize_qa_answers(
                row,
                preserve_unalignable=split != "train",
            )
            alignment_key = f"{split}:{lang}"
            stats = alignment_stats[alignment_key]
            stats["questions_seen"] += 1
            stats["answer_pairs_seen"] += answer_pair_count
            stats["corrected_answer_pairs"] += corrected_pair_count
            stats["unchanged_answer_pairs"] += max(0, answer_pair_count - corrected_pair_count) if alignment_status in {"aligned", "corrected_nearest"} else 0
            if alignment_status == "unalignable_eval":
                stats["unalignable_eval"] += 1
                alignment_reasons[alignment_key][alignment_reason or "unknown"] += 1
            elif alignment_status == "unalignable_train":
                stats["unalignable_train_skipped"] += 1
                alignment_reasons[alignment_key][alignment_reason or "unknown"] += 1
                rejected_rows[lang].append(
                    {
                        "id": str(row.get("id") or ""),
                        "title": row.get("title"),
                        "context": row.get("context"),
                        "question": row.get("question"),
                        "answers": row.get("answers"),
                        "answer_alignment_status": alignment_status,
                        "answer_alignment_reason": alignment_reason,
                        "language": lang,
                        "source_split": "train",
                        "source_revision": source["revision"],
                    }
                )
                continue
            elif alignment_status == "no_gold_answer":
                stats["no_gold_answer_questions"] += 1
                if split == "train":
                    stats["unalignable_train_skipped"] += 1
                    alignment_reasons[alignment_key][alignment_reason or "no_gold_answer"] += 1
                    rejected_rows[lang].append(
                        {
                            "id": str(row.get("id") or ""),
                            "title": row.get("title"),
                            "context": row.get("context"),
                            "question": row.get("question"),
                            "answers": row.get("answers"),
                            "answer_alignment_status": alignment_status,
                            "answer_alignment_reason": alignment_reason,
                            "language": lang,
                            "source_split": "train",
                            "source_revision": source["revision"],
                        }
                    )
                    continue
            elif alignment_status == "corrected_nearest":
                stats["corrected_questions"] += 1
            else:
                stats["exact_questions"] += 1
            features = encode_qa_example(
                tokenizer,
                normalized_row,
                lang,
                source["id"],
                source["revision"],
                "train",
                training=split == "train",
            )
            if not features:
                continue
            for feature in features:
                feature["window_type"] = "regular"
            if split == "dev":
                if dev_count >= dev_limit:
                    continue
                output_rows["dev"].extend(features)
                raw_key = f"{lang}_train_dev"
                dev_count += 1
            else:
                if train_count >= train_limit:
                    continue
                output_rows["train"].extend(features)
                raw_key = f"{lang}_train_train"
                train_count += 1
            feature_counts[f"{split}:{lang}"] += len(features)
            raw_key_id = (lang, "train", str(row.get("id") or features[0]["id"]))
            if raw_key_id not in raw_seen:
                raw_seen.add(raw_key_id)
                raw_rows[raw_key].append(
                    {
                    "id": features[0]["id"],
                    "title": row.get("title"),
                    "context": row.get("context"),
                    "question": row.get("question"),
                    "answers": normalized_row.get("answers"),
                    "source_answers_original": normalized_row.get("source_answers_original"),
                    "answer_alignment_status": normalized_row.get("answer_alignment_status"),
                    "answer_alignment_reason": normalized_row.get("answer_alignment_reason"),
                    "language": lang,
                    "split": split,
                    "source_revision": source["revision"],
                    }
                )
            if train_count >= train_limit and dev_count >= dev_limit:
                break
        counts[f"train:{lang}"] = train_count
        counts[f"dev:{lang}"] = dev_count

        public_eval_limit = targets["qa_eval"]
        validation_cache = raw_root(profile) / "qa" / f"{lang}_public_validation.jsonl"
        if profile == "full_67m":
            validation = ()
        else:
            validation = (
                list(iter_jsonl_file(validation_cache))
                if validation_cache.is_file()
                else get_stream(source["id"], config, "validation", source["revision"], shuffle=False)
            )
        eval_count = 0
        for row in validation:
            if public_eval_limit is not None and eval_count >= public_eval_limit:
                break
            normalized_row, alignment_status, alignment_reason, answer_pair_count, corrected_pair_count = normalize_qa_answers(
                row,
                preserve_unalignable=True,
            )
            alignment_key = f"test:{lang}"
            stats = alignment_stats[alignment_key]
            stats["questions_seen"] += 1
            stats["answer_pairs_seen"] += answer_pair_count
            stats["corrected_answer_pairs"] += corrected_pair_count
            stats["unchanged_answer_pairs"] += max(0, answer_pair_count - corrected_pair_count) if alignment_status in {"aligned", "corrected_nearest"} else 0
            if alignment_status == "unalignable_eval":
                stats["unalignable_eval"] += 1
                alignment_reasons[alignment_key][alignment_reason or "unknown"] += 1
            elif alignment_status == "no_gold_answer":
                stats["no_gold_answer_questions"] += 1
            elif alignment_status == "corrected_nearest":
                stats["corrected_questions"] += 1
            else:
                stats["exact_questions"] += 1
            features = encode_qa_example(tokenizer, normalized_row, lang, source["id"], source["revision"], "validation")
            if not features:
                continue
            for feature in features:
                feature["window_type"] = "regular"
            output_rows["test"].extend(features)
            feature_counts[f"test:{lang}"] += len(features)
            raw_key_id = (lang, "validation", str(row.get("id") or features[0]["id"]))
            if raw_key_id not in raw_seen:
                raw_seen.add(raw_key_id)
                raw_rows[f"{lang}_public_validation"].append(
                    {
                    "id": features[0]["id"],
                    "title": row.get("title"),
                    "context": row.get("context"),
                    "question": row.get("question"),
                    "answers": normalized_row.get("answers"),
                    "source_answers_original": normalized_row.get("source_answers_original"),
                    "answer_alignment_status": normalized_row.get("answer_alignment_status"),
                    "answer_alignment_reason": normalized_row.get("answer_alignment_reason"),
                    "language": lang,
                    "source_revision": source["revision"],
                    "source_split": "validation",
                    }
                )
            eval_count += 1
        if profile != "full_67m":
            counts[f"test:{lang}"] = eval_count
    long_rows = prepare_long_qa(tokenizer, profile, targets, output_rows["train"], manifest)
    for lang in ("en", "zh"):
        feature_counts[f"train:{lang}"] += sum(1 for row in long_rows if row["language"] == lang)
    for split, rows in output_rows.items():
        if profile == "full_67m" and split in {"dev", "test"}:
            continue
        path = out_root / f"qa_{split}.jsonl"
        dump_jsonl(path, rows)
        manifest["artifacts"][f"qa_{split}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "question_counts": {lang: counts[f"{split}:{lang}"] for lang in ("en", "zh")},
            "feature_counts": {lang: feature_counts[f"{split}:{lang}"] for lang in ("en", "zh")},
            "window_type_counts": {
                window_type: {lang: sum(1 for row in rows if row["language"] == lang and row.get("window_type") == window_type) for lang in ("en", "zh")}
                for window_type in ("regular", "long")
            },
            "input_length_counts": {
                "512": sum(1 for row in rows if len(row["input_ids"]) <= 512),
                "4096": sum(1 for row in rows if 3500 <= len(row["input_ids"]) <= 4096),
            },
            "regular_max_length": 512,
            "max_length": max((len(row["input_ids"]) for row in rows), default=512),
            "answer_alignment_counts": {
                status: sum(1 for row in rows if row.get("answer_alignment_status") == status)
                for status in ("aligned", "corrected_nearest", "unalignable_eval", "no_gold_answer")
            },
            "context_only_mask": "context_mask marks tokenizer sequence_id==1 positions; question and special tokens are zero",
            "offsets": "original context character start/end for every token; special and question tokens have [0,0]",
        }
    for key, rows in raw_rows.items():
        raw_path = raw_root(profile) / "qa" / f"{key}.jsonl"
        dump_jsonl(raw_path, rows)
        manifest["artifacts"][f"raw_qa_{key}"] = {
            "path": str(raw_path.relative_to(ROOT)),
            "rows": len(rows),
            "sha256": sha256_file(raw_path),
        }
    for lang, rows in rejected_rows.items():
        rejected_path = raw_root(profile) / "qa" / f"offset_rejected_{lang}.jsonl"
        dump_jsonl(rejected_path, rows)
        manifest["artifacts"][f"qa_offset_rejected_{lang}"] = {
            "path": str(rejected_path.relative_to(ROOT)),
            "rows": len(rows),
            "sha256": sha256_file(rejected_path),
            "language": lang,
            "reason_counts": {
                reason: sum(1 for row in rows if row.get("answer_alignment_reason") == reason)
                for reason in sorted({str(row.get("answer_alignment_reason") or "unknown") for row in rows})
            },
        }
    manifest["qa_answer_offset_alignment"] = {
        "policy": "Keep answer text unchanged. If the supplied span is not exact, correct only to a uniquely nearest exact occurrence. Skip unalignable training questions; preserve unalignable dev/public-eval questions with all span labels set to -100.",
        "input_source_by_language": input_sources,
        "counts_by_output_split_and_language": {
            key: dict(sorted(counts.items())) for key, counts in sorted(alignment_stats.items())
        },
        "reasons_by_output_split_and_language": {
            key: dict(sorted(counts.items())) for key, counts in sorted(alignment_reasons.items())
        },
        "unalignable_eval_questions": sum(stats["unalignable_eval"] for stats in alignment_stats.values()),
        "unalignable_training_questions_skipped": sum(stats["unalignable_train_skipped"] for stats in alignment_stats.values()),
    }


def read_gzip_prefix(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = []
    if not path.is_file():
        return rows
    try:
        with gzip.open(path, "rb") as compressed:
            while len(rows) < limit:
                line = compressed.readline()
                if not line:
                    break
                try:
                    rows.append(json.loads(line))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    break
    except (EOFError, OSError, zlib.error):
        # The last cached range can end mid-record or mid-deflate block; extend it below.
        return rows
    return rows


def ensure_gzip_prefix(path: Path, url: str, offset: int, file_size: int, chunk_size: int) -> int:
    end = min(offset + chunk_size - 1, file_size - 1)
    retries = 6
    for attempt in range(retries):
        current = path.stat().st_size if path.exists() else 0
        if current > end:
            return current
        start = current
        try:
            headers = {"Range": f"bytes={start}-{end}", "Accept-Encoding": "identity", "Cache-Control": "no-cache"}
            with requests.get(url, stream=True, headers=headers, timeout=(20, 90)) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206 for MLDR bytes {start}-{end}, got {response.status_code}")
                expected_range = f"bytes {start}-{end}/{file_size}"
                if response.headers.get("Content-Range") != expected_range:
                    raise RuntimeError(f"unexpected MLDR Content-Range {response.headers.get('Content-Range')!r}; expected {expected_range!r}")
                with path.open("ab" if start else "wb") as output:
                    for block in response.iter_content(chunk_size=256 * 1024):
                        if block:
                            output.write(block)
            current = path.stat().st_size
            if current < end + 1:
                raise RuntimeError(f"short MLDR prefix chunk: {current} bytes, expected at least {end + 1}")
            print(f"Cached MLDR prefix {path.name}: {current:,}/{file_size:,} bytes", flush=True)
            return current
        except Exception as exc:
            if attempt + 1 == retries:
                raise
            print(f"Retrying MLDR Range {start}-{end} ({attempt + 1}/{retries - 1}): {type(exc).__name__}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 16))
    return path.stat().st_size if path.exists() else 0


def iter_mldr_gzip(lang: str, split: str, limit: int | None = None, profile: str = "pilot") -> Iterator[dict[str, Any]]:
    source = DATASET_SOURCES["mldr"]
    filename = f"mldr-v1.0-{lang}/{split}.jsonl.gz"
    url = hf_hub_url(source["id"], filename=filename, revision=source["revision"], repo_type="dataset")
    if limit is None:
        raise ValueError("A finite row limit is required for the MLDR local-prefix workflow")
    if source["id"] not in MLDR_SOURCE_METADATA:
        info = HfApi().dataset_info(source["id"], revision=source["revision"], files_metadata=True)
        MLDR_SOURCE_METADATA[source["id"]] = {item.rfilename: item for item in info.siblings}
    sibling = MLDR_SOURCE_METADATA[source["id"]].get(filename)
    if sibling is None or sibling.size is None:
        raise RuntimeError(f"Pinned Hub metadata did not list {filename}")
    file_size = int(sibling.size)
    file_hash = sibling.lfs.sha256 if sibling.lfs else None
    prefix_path = raw_root(profile) / "mldr" / "prefixes" / f"{lang}_{split}.jsonl.gz.prefix"
    prefix_path.parent.mkdir(parents=True, exist_ok=True)
    if profile == "full_67m" and split == "train" and not prefix_path.exists():
        earlier_prefix = raw_root("full") / "mldr" / "prefixes" / prefix_path.name
        if earlier_prefix.is_file():
            shutil.copy2(earlier_prefix, prefix_path)
            print(f"Seeded resumable MLDR prefix from previous full profile: {prefix_path.name} ({prefix_path.stat().st_size:,} bytes)", flush=True)
    chunk_size = 1024 * 1024 if profile == "pilot" else 8 * 1024 * 1024
    available_rows: list[dict[str, Any]] = []
    while len(available_rows) < limit and (prefix_path.stat().st_size if prefix_path.exists() else 0) < file_size:
        available_rows = read_gzip_prefix(prefix_path, limit)
        if len(available_rows) >= limit:
            break
        current_size = prefix_path.stat().st_size if prefix_path.exists() else 0
        if current_size >= file_size:
            break
        # Download at most one profile-sized range, then check whether it contained enough rows.
        before = current_size
        ensure_gzip_prefix(prefix_path, url, current_size, file_size, chunk_size)
        if prefix_path.stat().st_size <= before:
            break
    available_rows = read_gzip_prefix(prefix_path, limit)
    prefix_size = prefix_path.stat().st_size
    MLDR_PREFIX_RECORDS[(profile, lang, split)] = {
        "source_id": source["id"],
        "source_revision": source["revision"],
        "source_filename": filename,
        "source_file_size_bytes": file_size,
        "source_file_sha256_from_hub_metadata": file_hash,
        "source_file_sha256_verified": False,
        "prefix_path": str(prefix_path.relative_to(ROOT)),
        "prefix_bytes_downloaded": prefix_size,
        "prefix_sha256": sha256_file(prefix_path),
        "rows_requested": limit,
        "rows_read": len(available_rows),
    }
    for row in available_rows:
        yield row


def encode_ids(tokenizer, text: str, max_length: int) -> list[int]:
    return tokenizer(text, truncation=True, max_length=max_length)["input_ids"]


def mldr_doc_id(passage: dict[str, Any], fallback: str) -> str:
    return str(passage.get("docid") or passage.get("id") or fallback)


def bm25_terms(text: str, lang: str) -> list[str]:
    if lang == "zh":
        characters = [char for char in text if not char.isspace() and char not in "，。！？；：、（）【】《》“”‘’"]
        if len(characters) < 2:
            return characters
        return [characters[i] + characters[i + 1] for i in range(len(characters) - 1)]
    return re.findall(r"[a-z0-9]+", text.casefold())


def build_bm25_index(documents: list[dict[str, Any]], lang: str) -> dict[str, Any]:
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = []
    for doc_index, doc in enumerate(documents):
        counts = Counter(bm25_terms(str(doc.get("text", "")), lang))
        lengths.append(sum(counts.values()))
        for term, term_count in counts.items():
            postings[term].append((doc_index, term_count))
    return {
        "postings": postings,
        "lengths": lengths,
        "average_length": sum(lengths) / max(1, len(lengths)),
        "document_count": len(documents),
    }


def bm25_rank(
    query: str,
    documents: list[dict[str, Any]],
    lang: str,
    top_k: int = 20,
    k1: float = 1.2,
    b: float = 0.75,
    index: dict[str, Any] | None = None,
) -> list[tuple[dict[str, Any], float]]:
    if not documents:
        return []
    index = index or build_bm25_index(documents, lang)
    postings = index["postings"]
    lengths = index["lengths"]
    avg_length = index["average_length"]
    query_terms = set(bm25_terms(query, lang))
    scores = [0.0] * len(documents)
    total_docs = index["document_count"]
    for term in query_terms:
        term_postings = postings.get(term)
        if not term_postings:
            continue
        df = len(term_postings)
        idf = math.log(1.0 + (total_docs - df + 0.5) / (df + 0.5))
        for doc_index, tf in term_postings:
            norm = tf + k1 * (1.0 - b + b * lengths[doc_index] / max(avg_length, 1.0))
            scores[doc_index] += idf * tf * (k1 + 1.0) / norm
    ordering = sorted(
        range(total_docs),
        key=lambda i: (-scores[i], str(documents[i].get("docid", documents[i].get("id", "")))),
    )[:top_k]
    return [(documents[index], scores[index]) for index in ordering]


def prepare_retrieval(tokenizer, profile: str, targets: dict[str, Any], manifest: dict[str, Any]) -> None:
    if profile == "full_67m":
        prepare_retrieval_full_67m(tokenizer, profile, targets, manifest)
        return
    out_root = artifact_root(profile)
    query_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "dev", "test")}
    raw_query_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    eval_positive_docs: dict[str, dict[str, dict[str, Any]]] = {"en": {}, "zh": {}}
    corpus_by_lang: dict[str, list[dict[str, Any]]] = {}
    for lang in ("en", "zh"):
        split_limits = {
            "train": targets["retrieval_train"],
            "dev": targets["retrieval_dev"],
            "test": targets["retrieval_test"],
        }
        for output_split, source_split in (("train", "train"), ("dev", "dev"), ("test", "test")):
            limit = split_limits[output_split]
            raw_count = 0
            for row in iter_mldr_gzip(lang, source_split, limit=limit, profile=profile):
                query_id = str(row.get("query_id") or f"{lang}:{source_split}:{raw_count}")
                query_text = str(row.get("query") or "")
                positives = row.get("positive_passages") or []
                source_negatives = row.get("negative_passages") or []
                if not query_text or not positives:
                    continue
                positives = [item for item in positives if item.get("text")]
                source_negatives = [item for item in source_negatives if item.get("text")]
                if not positives:
                    continue
                if output_split == "train" and not source_negatives:
                    continue
                positive = positives[0]
                # Stable hash ranking selects two hard negatives for the reranking derivation.
                selected_negatives = sorted(
                    source_negatives,
                    key=lambda item: hashlib.sha256(f"{SEED}:{query_id}:{mldr_doc_id(item, '')}".encode("utf-8")).hexdigest(),
                )[:2]
                qids = encode_ids(tokenizer, query_text, 128)
                positive_docid = mldr_doc_id(positive, f"{query_id}:positive:0")
                positive_cap = 8192 if output_split != "train" else _train_doc_cap(query_id, positive_docid)
                positive_ids = encode_ids(tokenizer, str(positive["text"]), positive_cap)
                negative_records = []
                for index, item in enumerate(selected_negatives):
                    doc_id = mldr_doc_id(item, f"{query_id}:negative:{index}")
                    cap = 8192 if output_split != "train" else _train_doc_cap(query_id, doc_id)
                    negative_records.append(
                        {
                            "docid": doc_id,
                            "text": str(item["text"]),
                            "input_ids": encode_ids(tokenizer, str(item["text"]), cap),
                            "max_length": cap,
                        }
                    )
                query_row = {
                    "query_ids": qids,
                    "positive_ids": positive_ids,
                    "negative_ids": negative_records[0]["input_ids"] if negative_records else [],
                    "language": lang,
                    "id": query_id,
                    "positive_docid": positive_docid,
                    "relevant_doc_ids": [mldr_doc_id(item, f"{query_id}:positive:{i}") for i, item in enumerate(positives)],
                    "negative_docid": negative_records[0]["docid"] if negative_records else None,
                    "negative_candidates": negative_records,
                    "query": query_text,
                    "positive_text": str(positive["text"]),
                    "source_id": DATASET_SOURCES["mldr"]["id"],
                    "source_revision": DATASET_SOURCES["mldr"]["revision"],
                    "source_split": source_split,
                    "document_window_policy": "train hash assigns maximum token-prefix cap 512/4096/8192 at 90/8/2 percent; evaluation uses the first 8192 tokens; natural short lengths retained",
                }
                query_rows[output_split].append(query_row)
                raw_query_rows[f"{lang}_{source_split}"].append(
                    {
                        "query_id": query_id,
                        "query": query_text,
                        "positive_passages": positives,
                        "negative_passages": source_negatives,
                        "source_revision": DATASET_SOURCES["mldr"]["revision"],
                        "source_split": source_split,
                    }
                )
                if output_split in ("dev", "test"):
                    for passage in positives:
                        docid = mldr_doc_id(passage, f"{query_id}:positive")
                        entry = eval_positive_docs[lang].setdefault(
                            docid,
                            {"docid": docid, "text": str(passage["text"]), "relevant_query_ids": []},
                        )
                        entry["relevant_query_ids"].append(query_id)
                raw_count += 1
            counts[f"{output_split}:{lang}"] = raw_count
    for split, rows in query_rows.items():
        path = out_root / f"retrieval_{split}.jsonl"
        dump_jsonl(path, rows)
        manifest["artifacts"][f"retrieval_{split}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "language_counts": {lang: counts[f"{split}:{lang}"] for lang in ("en", "zh")},
            "query_max_length": 128,
            "evaluation_document_cap": 8192,
        }
    for key, rows in raw_query_rows.items():
        raw_path = raw_root(profile) / "mldr" / f"{key}.jsonl"
        dump_jsonl(raw_path, rows)
        manifest["artifacts"][f"raw_mldr_{key}"] = {
            "path": str(raw_path.relative_to(ROOT)),
            "rows": len(rows),
            "sha256": sha256_file(raw_path),
            "revision": DATASET_SOURCES["mldr"]["revision"],
            "source_prefix": MLDR_PREFIX_RECORDS.get((profile, key.split("_", 1)[0], key.split("_", 1)[1])),
        }

    corpus_size = targets["retrieval_corpus"]
    candidate_limit = max(6000 if profile == "full" else 1000, corpus_size + 1000 if profile == "full" else corpus_size * 6)
    for lang in ("en", "zh"):
        candidates: dict[str, dict[str, Any]] = {}
        candidate_rows_read = 0
        for index, item in enumerate(iter_mldr_gzip(lang, "corpus", limit=candidate_limit, profile=profile)):
            candidate_rows_read += 1
            docid = mldr_doc_id(item, f"{lang}:corpus:{index}")
            if docid not in eval_positive_docs[lang]:
                candidates[docid] = {"docid": docid, "text": str(item.get("text") or "")}
        selected = list(eval_positive_docs[lang].values())
        selected_ids = {item["docid"] for item in selected}
        distractors = sorted(
            (item for item in candidates.values() if item["text"]),
            key=lambda item: hashlib.sha256(f"mldr-corpus-v1:{SEED}:{lang}:{item['docid']}".encode("utf-8")).hexdigest(),
        )
        selected.extend(distractors[: max(0, corpus_size - len(selected))])
        selected = selected[:corpus_size]
        corpus_rows = []
        raw_corpus_rows = []
        for item in selected:
            is_relevant = item["docid"] in selected_ids
            text = item["text"]
            input_ids = encode_ids(tokenizer, text, 8192)
            corpus_rows.append(
                {
                    "id": item["docid"],
                    "docid": item["docid"],
                    "input_ids": input_ids,
                    "text": text,
                    "language": lang,
                    "is_known_relevant": is_relevant,
                    "relevant_query_ids": item.get("relevant_query_ids", []),
                    "source_revision": DATASET_SOURCES["mldr"]["revision"],
                    "selection": "known dev/test positive" if is_relevant else "sha256-ranked distractor from source corpus prefix",
                }
            )
            raw_corpus_rows.append(
                {
                    "docid": item["docid"],
                    "text": text,
                    "language": lang,
                    "is_known_relevant": is_relevant,
                    "relevant_query_ids": item.get("relevant_query_ids", []),
                    "source_revision": DATASET_SOURCES["mldr"]["revision"],
                }
            )
        corpus_path = out_root / f"retrieval_corpus_{lang}.jsonl"
        dump_jsonl(corpus_path, corpus_rows)
        raw_corpus_path = raw_root(profile) / "mldr" / f"{lang}_research_corpus.jsonl"
        dump_jsonl(raw_corpus_path, raw_corpus_rows)
        manifest["artifacts"][f"retrieval_corpus_{lang}"] = {
            "path": str(corpus_path.relative_to(ROOT)),
            "sha256": sha256_file(corpus_path),
            "raw_path": str(raw_corpus_path.relative_to(ROOT)),
            "raw_sha256": sha256_file(raw_corpus_path),
            "rows": len(corpus_rows),
            "known_relevant_rows": sum(1 for item in corpus_rows if item["is_known_relevant"]),
            "distractor_rows": sum(1 for item in corpus_rows if not item["is_known_relevant"]),
            "candidate_source_prefix_rows_examined": candidate_rows_read,
            "source_prefix": MLDR_PREFIX_RECORDS.get((profile, lang, "corpus")),
            "corpus_selection": "include all unique dev/test positives, then fill to target by ascending SHA-256 rank of source corpus docid among the examined source corpus prefix",
            "language": lang,
        }
        corpus_by_lang[lang] = corpus_rows

    combined_corpus = []
    for lang in ("en", "zh"):
        corpus_path = out_root / f"retrieval_corpus_{lang}.jsonl"
        with corpus_path.open("r", encoding="utf-8") as stream:
            combined_corpus.extend(json.loads(line) for line in stream if line.strip())
    combined_corpus_path = out_root / "retrieval_corpus.jsonl"
    dump_jsonl(combined_corpus_path, combined_corpus)
    manifest["artifacts"]["retrieval_corpus"] = {
        "path": str(combined_corpus_path.relative_to(ROOT)),
        "sha256": sha256_file(combined_corpus_path),
        "rows": len(combined_corpus),
        "language_counts": {lang: sum(1 for row in combined_corpus if row["language"] == lang) for lang in ("en", "zh")},
        "interface_fields": ["id", "input_ids", "language"],
    }

    for split in ("dev", "test"):
        eval_query_rows = []
        for row in query_rows[split]:
            eval_query_rows.append(
                {
                    "id": row["id"],
                    "input_ids": row["query_ids"],
                    "language": row["language"],
                    "relevant_doc_ids": row["relevant_doc_ids"],
                    "query": row["query"],
                    "source_revision": row["source_revision"],
                }
            )
        eval_path = out_root / f"retrieval_queries_{split}.jsonl"
        dump_jsonl(eval_path, eval_query_rows)
        manifest["artifacts"][f"retrieval_queries_{split}"] = {
            "path": str(eval_path.relative_to(ROOT)),
            "sha256": sha256_file(eval_path),
            "rows": len(eval_query_rows),
            "language_counts": {lang: sum(1 for row in eval_query_rows if row["language"] == lang) for lang in ("en", "zh")},
            "interface_fields": ["id", "input_ids", "language", "relevant_doc_ids"],
        }

    rerank_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in ("train", "dev", "test")}
    bm25_indices = {lang: build_bm25_index(corpus_by_lang[lang], lang) for lang in ("en", "zh")}
    manifest["artifacts"]["bm25_index"] = {
        "status": "built_in_memory",
        "tokenization": "lowercase alphanumeric words for English; non-space character bigrams for Chinese",
        "k1": 1.2,
        "b": 0.75,
        "documents_by_language": {lang: len(corpus_by_lang[lang]) for lang in ("en", "zh")},
    }
    for row in query_rows["train"]:
        negatives = row["negative_candidates"]
        if len(negatives) < 2:
            continue
        candidate_texts = [row["positive_text"], negatives[0]["text"], negatives[1]["text"]]
        rerank_rows["train"].append(
            {
                "id": row["id"],
                "language": row["language"],
                "query_ids": row["query_ids"],
                "candidate_ids": [encode_ids(tokenizer, text, 1024) for text in candidate_texts],
                "labels": [1, 0, 0],
                "candidate_docids": [row["positive_docid"], negatives[0]["docid"], negatives[1]["docid"]],
                "query": row["query"],
                "candidate_texts": candidate_texts,
                "source_id": DATASET_SOURCES["mldr"]["id"],
                "source_revision": DATASET_SOURCES["mldr"]["revision"],
            }
        )
    for split in ("dev", "test"):
        for row in query_rows[split]:
            ranked = bm25_rank(
                row["query"],
                corpus_by_lang[row["language"]],
                row["language"],
                top_k=20,
                index=bm25_indices[row["language"]],
            )
            if not ranked:
                continue
            candidates = [item for item, _score in ranked]
            rerank_rows[split].append(
                {
                    "id": row["id"],
                    "language": row["language"],
                    "query_ids": row["query_ids"],
                    "candidate_ids": [item["input_ids"][:1024] for item in candidates],
                    "labels": [1 if item["docid"] in set(row["relevant_doc_ids"]) else 0 for item in candidates],
                    "candidate_docids": [item["docid"] for item in candidates],
                    "bm25_scores": [float(score) for _item, score in ranked],
                    "relevant_doc_ids": row["relevant_doc_ids"],
                    "query": row["query"],
                    "candidate_texts": [item["text"] for item in candidates],
                    "source_id": DATASET_SOURCES["mldr"]["id"],
                    "source_revision": DATASET_SOURCES["mldr"]["revision"],
                    "candidate_source": "fixed BM25 top-20 over retrieval_corpus.jsonl",
                }
            )
    for split, rows in rerank_rows.items():
        path = out_root / f"reranking_{split}.jsonl"
        dump_jsonl(path, rows)
        manifest["artifacts"][f"reranking_{split}"] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": len(rows),
            "language_counts": {lang: sum(1 for row in rows if row["language"] == lang) for lang in ("en", "zh")},
            "candidates_per_group": 3 if split == "train" else 20,
            "candidate_source": "two deterministic MLDR hard negatives" if split == "train" else "fixed BM25 top-20 over reduced corpus",
        }


def prepare_retrieval_full_67m(tokenizer, profile: str, targets: dict[str, Any], manifest: dict[str, Any]) -> None:
    out_root = artifact_root(profile)
    query_rows: list[dict[str, Any]] = []
    raw_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    for lang in ("en", "zh"):
        raw_count = 0
        for row in iter_mldr_gzip(lang, "train", limit=targets["retrieval_train"], profile=profile):
            query_id = str(row.get("query_id") or f"{lang}:train:{raw_count}")
            query_text = str(row.get("query") or "")
            positives = [item for item in (row.get("positive_passages") or []) if item.get("text")]
            source_negatives = [item for item in (row.get("negative_passages") or []) if item.get("text")]
            if not query_text or not positives or len(source_negatives) < 2:
                continue
            positive = positives[0]
            selected_negatives = sorted(
                source_negatives,
                key=lambda item: hashlib.sha256(f"{SEED}:{query_id}:{mldr_doc_id(item, '')}".encode("utf-8")).hexdigest(),
            )[:2]
            qids = encode_ids(tokenizer, query_text, 128)
            positive_docid = mldr_doc_id(positive, f"{query_id}:positive:0")
            positive_cap = _train_doc_cap(query_id, positive_docid)
            positive_ids = encode_ids(tokenizer, str(positive["text"]), positive_cap)
            negative_records = []
            for index, item in enumerate(selected_negatives):
                doc_id = mldr_doc_id(item, f"{query_id}:negative:{index}")
                cap = _train_doc_cap(query_id, doc_id)
                negative_records.append(
                    {
                        "docid": doc_id,
                        "text": str(item["text"]),
                        "input_ids": encode_ids(tokenizer, str(item["text"]), cap),
                        "max_length": cap,
                    }
                )
            query_rows.append(
                {
                    "query_ids": qids,
                    "positive_ids": positive_ids,
                    "negative_ids": negative_records[0]["input_ids"],
                    "language": lang,
                    "id": query_id,
                    "positive_docid": positive_docid,
                    "relevant_doc_ids": [mldr_doc_id(item, f"{query_id}:positive:{i}") for i, item in enumerate(positives)],
                    "negative_docid": negative_records[0]["docid"],
                    "negative_candidates": negative_records,
                    "query": query_text,
                    "positive_text": str(positive["text"]),
                    "source_id": DATASET_SOURCES["mldr"]["id"],
                    "source_revision": DATASET_SOURCES["mldr"]["revision"],
                    "source_split": "train",
                    "document_window_policy": "train hash assigns maximum token-prefix cap 512/4096/8192 at 90/8/2 percent; evaluation uses the first 8192 tokens; natural short lengths retained",
                }
            )
            raw_rows[f"{lang}_train"].append(
                {
                    "query_id": query_id,
                    "query": query_text,
                    "positive_passages": positives,
                    "negative_passages": source_negatives,
                    "source_revision": DATASET_SOURCES["mldr"]["revision"],
                    "source_split": "train",
                }
            )
            raw_count += 1
        counts[f"train:{lang}"] = raw_count

    train_path = out_root / "retrieval_train.jsonl"
    dump_jsonl(train_path, query_rows)
    manifest["artifacts"]["retrieval_train"] = {
        "path": str(train_path.relative_to(ROOT)),
        "sha256": sha256_file(train_path),
        "rows": len(query_rows),
        "language_counts": {lang: counts[f"train:{lang}"] for lang in ("en", "zh")},
        "query_max_length": 128,
        "document_length_proportions": {"512": 0.9, "4096": 0.08, "8192": 0.02},
        "document_window_policy": "hash-assigned maximum token-prefix cap; first-token truncation",
        "evaluation_data_profile": "full",
    }
    for key, rows in raw_rows.items():
        raw_path = raw_root(profile) / "mldr" / f"{key}.jsonl"
        dump_jsonl(raw_path, rows)
        manifest["artifacts"][f"raw_mldr_{key}"] = {
            "path": str(raw_path.relative_to(ROOT)),
            "rows": len(rows),
            "sha256": sha256_file(raw_path),
            "revision": DATASET_SOURCES["mldr"]["revision"],
            "source_prefix": MLDR_PREFIX_RECORDS.get((profile, key.split("_", 1)[0], "train")),
        }

    old_manifest = json.loads((artifact_root("full") / "manifest_retrieval.json").read_text(encoding="utf-8"))
    for key in (
        "retrieval_dev",
        "retrieval_test",
        "retrieval_corpus",
        "retrieval_queries_dev",
        "retrieval_queries_test",
        "reranking_dev",
        "reranking_test",
    ):
        reuse_manifest_artifact("full", profile, "retrieval", key, manifest)
    for lang in ("en", "zh"):
        for split in ("dev", "test"):
            reuse_manifest_artifact("full", profile, "retrieval", f"raw_mldr_{lang}_{split}", manifest)
    manifest["artifacts"]["bm25_index"] = dict(old_manifest["artifacts"]["bm25_index"])
    manifest["artifacts"]["bm25_index"]["reused_from_profile"] = "full"

    rerank_rows = []
    for row in query_rows:
        negatives = row["negative_candidates"]
        candidate_texts = [row["positive_text"], negatives[0]["text"], negatives[1]["text"]]
        rerank_rows.append(
            {
                "id": row["id"],
                "language": row["language"],
                "query_ids": row["query_ids"],
                "candidate_ids": [encode_ids(tokenizer, text, 1024) for text in candidate_texts],
                "labels": [1, 0, 0],
                "candidate_docids": [row["positive_docid"], negatives[0]["docid"], negatives[1]["docid"]],
                "query": row["query"],
                "candidate_texts": candidate_texts,
                "source_id": DATASET_SOURCES["mldr"]["id"],
                "source_revision": DATASET_SOURCES["mldr"]["revision"],
            }
        )
    path = out_root / "reranking_train.jsonl"
    dump_jsonl(path, rerank_rows)
    manifest["artifacts"]["reranking_train"] = {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "rows": len(rerank_rows),
        "language_counts": {lang: sum(1 for row in rerank_rows if row["language"] == lang) for lang in ("en", "zh")},
        "candidates_per_group": 3,
        "candidate_source": "two deterministic MLDR hard negatives",
    }


def _train_doc_cap(query_id: str, doc_id: str) -> int:
    bucket = stable_int(f"mldr-length-v1:{SEED}:{query_id}:{doc_id}") % 100
    return 512 if bucket < 90 else 4096 if bucket < 98 else 8192


def run(
    profile: str,
    only: list[str],
    tokenizer_path: Path,
    qa_train: int | None,
    qa_dev: int | None,
    retrieval_train: int | None,
    classification_train: int | None,
    plan: dict[str, Any],
    plan_path: Path,
    plan_bytes: bytes,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, use_fast=True)
    if not tokenizer.is_fast:
        raise RuntimeError("A fast tokenizer is required to preserve QA offsets and context masks")
    targets = profile_targets(profile, qa_train, qa_dev, retrieval_train, classification_train, plan)
    manifest = manifest_template(profile, tokenizer_path)
    manifest["data_profile"] = profile
    plan_relative, plan_snapshot_relative = save_plan_snapshot(profile, plan_path, plan_bytes)
    manifest["experiment_plan"] = {
        "path": plan_relative,
        "sha256": sha256_bytes(plan_bytes),
        "snapshot_path": plan_snapshot_relative,
        "profile_id": plan.get("data_profile"),
    }
    manifest["data_targets"] = targets
    if "wiki" in only:
        prepare_wiki(tokenizer, profile, manifest, targets)
    if "classification" in only:
        prepare_classification(tokenizer, profile, targets, manifest)
    if "qa" in only:
        prepare_qa(tokenizer, profile, targets, manifest)
    if "retrieval" in only:
        prepare_retrieval(tokenizer, profile, targets, manifest)
    filename = f"manifest_{only[0]}.json" if len(only) == 1 else "manifest.json"
    path = save_manifest(profile, manifest, filename)
    print(f"[{profile}] Manifest saved to {path.relative_to(ROOT)}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("pilot", "full", "full_67m"), default="pilot")
    parser.add_argument("--only", default="wiki,classification,qa,retrieval", help="Comma-separated task names")
    parser.add_argument("--tokenizer-path", type=Path, default=TOKENIZER_DIR)
    parser.add_argument("--plan", type=Path, default=ROOT / "configs" / "experiment_plan.json")
    parser.add_argument("--classification-train", type=int)
    parser.add_argument("--qa-train", type=int)
    parser.add_argument("--qa-dev", type=int)
    parser.add_argument("--retrieval-train", type=int)
    parser.add_argument("--merge-manifest", action="store_true", help="Merge independent manifest_{task}.json files into manifest.json")
    parser.add_argument("--split-existing-wiki", action="store_true", help="Separate already-prepared combined Wiki rows into train/dev/test files without reacquiring source data")
    args = parser.parse_args()
    if args.split_existing_wiki:
        path = split_existing_wiki_files(args.profile)
        print(f"[{args.profile}] Train/holdout Wiki arrays separated; manifest saved to {path.relative_to(ROOT)}", flush=True)
        return 0
    if args.merge_manifest:
        path = merge_task_manifests(args.profile)
        print(f"[{args.profile}] Combined manifest saved to {path.relative_to(ROOT)}", flush=True)
        return 0
    only = [item.strip() for item in args.only.split(",") if item.strip()]
    valid = {"wiki", "classification", "qa", "retrieval"}
    if not only or any(item not in valid for item in only):
        parser.error(f"--only values must be comma-separated members of {sorted(valid)}")
    if any(value is not None and value < 1 for value in (args.classification_train, args.qa_train, args.qa_dev, args.retrieval_train)):
        parser.error("sample count options must be positive")
    plan_path = args.plan if args.plan.is_absolute() else ROOT / args.plan
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    run(
        args.profile,
        only,
        args.tokenizer_path,
        args.qa_train,
        args.qa_dev,
        args.retrieval_train,
        args.classification_train,
        plan,
        plan_path,
        plan_bytes,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
