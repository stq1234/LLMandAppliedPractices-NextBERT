#!/usr/bin/env python
"""Download pinned model and dataset-card assets into this project."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "jhu-clsp/mmBERT-base"
MODEL_REVISION = "8c1f42296af5de623b0ad6d3a5ed6a5a1838d82c"
MODEL_FILES = [
    ".gitattributes",
    "README.md",
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]
DATASETS = {
    "wikipedia": {
        "id": "wikimedia/wikipedia",
        "revision": "b04c8d1ceb2f5cd4588862100d08de323dccfbaa",
        "license": "CC-BY-SA-3.0 and GFDL (Hub dataset card)",
        "configs": ["20231101.zh", "20231101.en"],
    },
    "xnli": {
        "id": "facebook/xnli",
        "revision": "b8dd5d7af51114dbda02c0e3f6133f332186418e",
        "license": "not declared in the pinned Hub dataset metadata",
        "configs": ["zh", "en"],
    },
    "squad": {
        "id": "rajpurkar/squad",
        "revision": "7b6d24c440a36b6815f21b70d25016731768db1f",
        "license": "CC-BY-SA-4.0 (Hub dataset card)",
        "configs": ["plain_text"],
    },
    "cmrc2018": {
        "id": "hfl/cmrc2018",
        "revision": "137f2c45a24275fb68f6961c4d357f46288886aa",
        "license": "CC-BY-SA-4.0 (Hub dataset card)",
        "configs": ["default"],
    },
    "mldr": {
        "id": "Shitao/MLDR",
        "revision": "d67138e705d963e346253a80e59676ddb418810a",
        "license": "MIT (Hub dataset card)",
        "configs": ["mldr-v1.0-zh", "mldr-v1.0-en"],
    },
}


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(json.dumps(portable_paths(obj), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp_path.replace(path)


def portable_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: portable_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [portable_paths(item) for item in value]
    if isinstance(value, str) and value.startswith(("artifacts\\", "data\\", "src\\", "configs\\")):
        return value.replace("\\", "/")
    return value


def download_range_part(
    url: str,
    part_path: Path,
    byte_start: int,
    byte_end: int,
    total_size: int,
    timeout: tuple[int, int],
    retries: int = 3,
) -> Path:
    expected = byte_end - byte_start + 1
    part_path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        existing = part_path.stat().st_size if part_path.exists() else 0
        if existing > expected:
            part_path.unlink()
            existing = 0
        if existing == expected:
            return part_path
        request_start = byte_start + existing
        headers = {
            "Range": f"bytes={request_start}-{byte_end}",
            "Accept-Encoding": "identity",
            "User-Agent": "nextbert-local-asset-preparer/1.0",
        }
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as response:
                if response.status_code != 206:
                    raise RuntimeError(f"expected HTTP 206 for bytes {request_start}-{byte_end}, got {response.status_code}")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                expected_range = (request_start, byte_end, total_size)
                if not match or tuple(map(int, match.groups())) != expected_range:
                    raise RuntimeError(
                        f"unexpected Content-Range {response.headers.get('Content-Range')!r}; "
                        f"expected bytes {request_start}-{byte_end}/{total_size}"
                    )
                mode = "ab" if existing else "wb"
                with part_path.open(mode) as output:
                    for block in response.iter_content(chunk_size=1024 * 1024):
                        if block:
                            output.write(block)
            if part_path.stat().st_size != expected:
                raise RuntimeError(f"short range part {part_path.name}: {part_path.stat().st_size}/{expected} bytes")
            return part_path
        except Exception as exc:
            if attempt + 1 == retries:
                raise
            print(
                f"Retrying model range {request_start}-{byte_end} ({attempt + 1}/{retries - 1}): {type(exc).__name__}: {exc}",
                flush=True,
            )
            time.sleep(min(2 ** attempt, 16))
    return part_path


def download_model_weight(workers: int = 16, chunk_mib: int = 8) -> dict[str, Any]:
    output_dir = ROOT / "artifacts" / "mmbert-base"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "pytorch_model.bin"
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION, files_metadata=True)
    weight = next((file for file in info.siblings if file.rfilename == "pytorch_model.bin"), None)
    if weight is None or weight.size is None or not weight.lfs or not weight.lfs.sha256:
        raise RuntimeError("Pinned Hub metadata lacks the pytorch_model.bin size or SHA-256")
    size = int(weight.size)
    expected_hash = weight.lfs.sha256
    if model_path.exists() and model_path.stat().st_size == size and sha256_file(model_path) == expected_hash:
        return {"path": model_path, "size": size, "sha256": expected_hash, "status": "already_verified"}

    part_dir = output_dir / ".download-parts" / "pytorch_model.bin"
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = chunk_mib * 1024 * 1024
    ranges = [(start, min(start + chunk_size, size) - 1) for start in range(0, size, chunk_size)]
    url = hf_hub_url(MODEL_ID, filename="pytorch_model.bin", revision=MODEL_REVISION, repo_type="model")
    t0 = time.monotonic()
    completed = 0
    completed_bytes = 0
    print(f"Downloading pinned model weight: {size:,} bytes in {len(ranges)} resumable ranges ({workers} workers).", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_to_range = {
            pool.submit(
                download_range_part,
                url,
                part_dir / f"{idx:05d}.part",
                start,
                end,
                size,
                (20, 60),
            ): (idx, start, end)
            for idx, (start, end) in enumerate(ranges)
        }
        for future in as_completed(future_to_range):
            idx, start, end = future_to_range[future]
            future.result()
            completed += 1
            completed_bytes += end - start + 1
            elapsed = max(time.monotonic() - t0, 0.01)
            rate = completed_bytes / elapsed / (1024 * 1024)
            print(f"  ranges {completed}/{len(ranges)}; verified bytes staged {completed_bytes:,}/{size:,}; avg {rate:.2f} MiB/s", flush=True)

    temp_model = output_dir / "pytorch_model.bin.tmp"
    digest = hashlib.sha256()
    written = 0
    with temp_model.open("wb") as output:
        for idx in range(len(ranges)):
            part_path = part_dir / f"{idx:05d}.part"
            with part_path.open("rb") as part:
                for block in iter(lambda: part.read(8 * 1024 * 1024), b""):
                    output.write(block)
                    digest.update(block)
                    written += len(block)
    if written != size or digest.hexdigest() != expected_hash:
        temp_model.unlink(missing_ok=True)
        raise RuntimeError(
            f"Model verification failed: size={written}/{size}, sha256={digest.hexdigest()}, expected={expected_hash}"
        )
    temp_model.replace(model_path)
    shutil.rmtree(part_dir)
    try:
        (output_dir / ".download-parts").rmdir()
    except OSError:
        pass
    return {"path": model_path, "size": size, "sha256": expected_hash, "status": "downloaded_and_verified"}


def download_small_assets() -> dict[str, Any]:
    local_dir = ROOT / "artifacts" / "mmbert-base"
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(local_dir),
        allow_patterns=MODEL_FILES,
        max_workers=4,
    )
    result: dict[str, Any] = {}
    for filename in MODEL_FILES:
        path = local_dir / filename
        if not path.is_file():
            raise RuntimeError(f"Missing model asset {path}")
        result[filename] = {"size_bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def download_source_cards() -> list[dict[str, Any]]:
    card_dir = ROOT / "artifacts" / "raw" / "source_cards"
    card_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for key, source in DATASETS.items():
        repo = source["id"]
        revision = source["revision"]
        try:
            filename = hf_hub_download(
                repo_id=repo,
                filename="README.md",
                revision=revision,
                repo_type="dataset",
                local_dir=str(card_dir / key),
            )
            card_path = Path(filename)
            card_sha = sha256_file(card_path)
            card_state = "downloaded"
        except Exception as exc:
            card_path = card_dir / key / "README.md"
            card_sha = None
            card_state = f"unavailable: {type(exc).__name__}: {exc}"
        records.append(
            {
                "key": key,
                "dataset_id": repo,
                "revision": revision,
                "license_status": source["license"],
                "config_names": source["configs"],
                "card_path": str(card_path.relative_to(ROOT)) if card_sha else None,
                "card_sha256": card_sha,
                "card_status": card_state,
            }
        )
    return records


def write_asset_manifest(weight_record: dict[str, Any], model_files: dict[str, Any], cards: list[dict[str, Any]]) -> Path:
    records = {
        "manifest_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "license": "MIT (pinned model card)",
            "weight": {
                "path": str(Path(weight_record["path"]).relative_to(ROOT)),
                "size_bytes": weight_record["size"],
                "sha256": weight_record["sha256"],
                "status": weight_record["status"],
            },
            "files": model_files,
        },
        "datasets": cards,
    }
    path = ROOT / "artifacts" / "assets_manifest.json"
    atomic_json(path, records)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=8)
    parser.add_argument("--skip-cards", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.chunk_mib < 1:
        parser.error("--workers and --chunk-mib must be positive")
    model_files = download_small_assets()
    print("Small model files verified.", flush=True)
    weight_record = download_model_weight(args.workers, args.chunk_mib)
    print(f"Model weight {weight_record['status']}: {weight_record['size']:,} bytes; SHA-256 {weight_record['sha256']}", flush=True)
    cards = [] if args.skip_cards else download_source_cards()
    manifest_path = write_asset_manifest(weight_record, model_files, cards)
    print(f"Asset manifest: {manifest_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
