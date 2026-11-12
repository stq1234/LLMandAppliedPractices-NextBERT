"""Reaggregate saved QA predictions on CPU, retaining the original result bytes."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def correct_result(path: Path, execute: bool = False) -> dict:
    original = path.read_bytes()
    payload = json.loads(original)
    if payload.get("task") != "qa" or payload.get("evaluation_scope") != "prepared_data_scope":
        raise ValueError(f"unexpected QA result: {path}")
    rows = payload["predictions"]
    if not rows or len(rows) != payload["question_count"]:
        raise ValueError(f"QA predictions/count mismatch: {path}")
    groups = defaultdict(list)
    for row in rows:
        if row["language"] not in {"en", "zh"}:
            raise ValueError(f"unexpected QA language: {path}")
        if any(not math.isfinite(float(row[key])) or not 0 <= float(row[key]) <= 1
               for key in ("f1", "exact_match")):
            raise ValueError(f"invalid per-question score: {path}")
        groups[row["language"]].append(row)
    if set(groups) != {"en", "zh"}:
        raise ValueError(f"missing QA language: {path}")
    corrected = {key + "_by_language": {
        language: {key: statistics.mean(float(row[key]) for row in values), "count": len(values)}
        for language, values in sorted(groups.items())
    } for key in ("exact_match", "f1")}
    for key in ("exact_match", "f1"):
        if not math.isclose(float(payload[key]), statistics.mean(float(row[key]) for row in rows),
                            rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"overall score differs from saved predictions: {path}/{key}")
    changed = any(payload.get(key) != value for key, value in corrected.items())
    summary = {"path": str(path), "changed": changed, "execute": execute,
               "f1_by_language": corrected["f1_by_language"]}
    if not execute or not changed:
        return summary
    backup = path.with_name(path.stem + ".before_language_f1_fix.json")
    if backup.exists():
        raise FileExistsError(f"original result snapshot already exists: {backup}")
    if path.read_bytes() != original:
        raise RuntimeError(f"QA result changed during correction: {path}")
    with backup.open("xb") as stream:
        stream.write(original)
    payload.update(corrected)
    payload["language_metric_correction"] = {
        "scope": "CPU arithmetic mean of saved per-question scores; predictions unchanged",
        "original_result_sha256": hashlib.sha256(original).hexdigest(),
        "original_result_snapshot": str(backup),
        "corrected_utc": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_suffix(".cpu_fix.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    evaluations = ROOT / "results" / "formal_67m"
    paths = [evaluations / "evaluation" / "tasks" / variant / f"qa_{split}.json"
             for variant in ("V0", "V3") for split in ("dev", "test")]
    paths += [evaluations / directory / "tasks" / variant / f"qa_{split}.json"
              for directory, variant in (("evaluation_gpu0_v1", "V1"), ("evaluation_gpu7_v2", "V2"))
              for split in ("dev", "test")]
    for path in paths:
        print(json.dumps(correct_result(path, args.execute), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
