"""Run the reproducible CPU-only NextGenBERT configuration checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nextbert_lab import collect_evidence, write_evidence


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-8k", action="store_true", help="skip the optional tiny 8K CPU forward pass")
    parser.add_argument("--output-dir", type=Path, default=package_root / "results")
    args = parser.parse_args()

    evidence = collect_evidence(run_long_probe=not args.skip_8k)
    write_evidence(evidence, args.output_dir)
    print(json.dumps({
        "status": "completed",
        "device_used": evidence["environment"]["device_used"],
        "threads": evidence["environment"]["torch_threads"],
        "mmbert_parameters": evidence["selected_pretrained_route"]["total_parameters"],
        "scratch_estimate": evidence["scratch_alternative"]["estimated_total_parameters"],
        "functional_checks": "passed",
        "8k_probe": evidence["tiny_8k_runtime_probe"]["status"],
        "json": str(args.output_dir / "verification.json"),
        "csv": str(args.output_dir / "parameter_counts.csv"),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
