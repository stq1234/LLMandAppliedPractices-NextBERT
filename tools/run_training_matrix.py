"""Print plan-driven training commands; run only with explicit --execute."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from train_nextbert import adaptation_stage_schedule, qa_phase_steps  # noqa: E402


def _load_plan() -> dict[str, Any]:
    return json.loads((ROOT / "configs/experiment_plan.json").read_text(encoding="utf-8"))


def _default_data_dir(plan: Mapping[str, Any]) -> Path:
    profile = str(plan.get("data_profile", "full"))
    return ROOT / "data" / "prepared" / profile


def _should_delete_resumed_checkpoint(variant: str, resume_path: Path,
                                     shared_prefix_checkpoint: Path | None) -> bool:
    if variant == "V2" and shared_prefix_checkpoint is not None:
        if resume_path.resolve() == shared_prefix_checkpoint.resolve():
            return False
    return True


def build_commands(model_dir: Path, data_dir: Path | None, output: Path,
                   plan: Mapping[str, Any] | None = None) -> list[list[str]]:
    plan = plan or _load_plan()
    data_dir = data_dir or _default_data_dir(plan)
    train = ROOT / "src/train_nextbert.py"
    stages = adaptation_stage_schedule(plan)
    total_steps = int(plan["adaptation"].get("full_branch_steps", stages[-1]["end_step"]))
    shared_steps = int(plan["adaptation"].get("shared_prefix", {}).get("steps", 0))
    if total_steps != int(stages[-1]["end_step"]):
        raise ValueError("adaptation.full_branch_steps must equal the sum of stage optimizer_steps")
    if shared_steps and shared_steps not in {int(stage["end_step"]) for stage in stages}:
        raise ValueError("shared_prefix.steps must coincide with an MLM stage boundary")

    commands: list[list[str]] = []
    final: dict[str, Path] = {}
    shared_checkpoint: Path | None = None
    for variant in ("V0", "V1", "V2", "V3"):
        previous = shared_checkpoint if variant == "V3" else None
        for stage in stages:
            length = int(stage["max_length"])
            start_step = int(stage["start_step"])
            absolute_step = int(stage["end_step"])
            if variant == "V3" and absolute_step <= shared_steps:
                continue
            stage_out = output / "mlm" / variant / str(length)
            cmd = [sys.executable, str(train), "--task", "mlm", "--variant", variant,
                   "--model-dir", str(model_dir), "--data", str(data_dir / f"wiki_{length}.npy"),
                   "--length", str(length), "--steps", str(absolute_step),
                   "--schedule-total-steps", str(total_steps), "--seed", str(plan["adaptation"]["seed"]),
                   "--output-dir", str(stage_out)]
            if previous is not None:
                cmd += ["--resume-from", str(previous)]
            if variant == "V3" and start_step >= shared_steps:
                cmd += ["--mode", "balanced"]
                if shared_checkpoint is not None and previous == shared_checkpoint:
                    cmd += ["--shared-prefix-resume"]
            commands.append(cmd)
            previous = stage_out / f"{variant}_mlm_checkpoint.pt"
            if variant == "V2" and absolute_step == shared_steps:
                shared_checkpoint = previous
        if previous is None:
            raise ValueError(f"no MLM stages were generated for {variant}")
        final[variant] = previous

    downstream = plan["downstream"]
    qa_steps = qa_phase_steps(plan)
    task_steps = {
        "classification": int(downstream["classification"]["steps"]),
        "qa": qa_steps["regular"] + qa_steps["long"],
        "retrieval": int(downstream["retrieval"]["steps"]),
        "reranking": int(downstream["reranking"]["steps"]),
    }
    for group in downstream["run_matrix"]:
        for variant in group["variants"]:
            for task in group["tasks"]:
                for seed in group["seeds"]:
                    data_task = "retrieval" if task == "reranking" else task
                    cmd = [sys.executable, str(train), "--task", task, "--variant", variant,
                           "--model-dir", str(model_dir), "--data", str(data_dir / f"{data_task}_train.jsonl"),
                           "--steps", str(task_steps[task]), "--seed", str(seed), "--init-from", str(final[variant]),
                           "--final-weights-only",
                           "--output-dir", str(output / "downstream" / variant / task / str(seed))]
                    if task == "retrieval":
                        cmd += ["--physical-document-batch",
                                str(int(downstream["retrieval"]["physical_document_batch"]))]
                    commands.append(cmd)
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=ROOT / "artifacts/mmbert-base")
    parser.add_argument("--data-dir", type=Path,
                        help="prepared profile directory; defaults to experiment_plan.json data_profile")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/formal")
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = _load_plan()
    data_dir = args.data_dir or _default_data_dir(plan)
    commands = build_commands(args.model_dir, data_dir, args.output_dir, plan)
    if args.execute:
        needed = {Path(cmd[cmd.index("--data") + 1]) for cmd in commands}
        missing = [str(path) for path in needed if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Prepared training files missing: {missing}")
        if args.output_dir.exists() and any(args.output_dir.rglob("*checkpoint.pt")):
            raise FileExistsError("Output contains checkpoints; use a fresh directory or resume individual stages explicitly")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    shared_steps = int(plan["adaptation"].get("shared_prefix", {}).get("steps", 0))
    shared_stage = next((stage for stage in adaptation_stage_schedule(plan)
                         if int(stage["end_step"]) == shared_steps), None)
    shared_prefix_checkpoint = (args.output_dir / "mlm" / "V2" / str(shared_stage["max_length"])
                                / "V2_mlm_checkpoint.pt") if shared_stage else None
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            if "--resume-from" in cmd:
                previous = Path(cmd[cmd.index("--resume-from") + 1]).resolve()
                variant = cmd[cmd.index("--variant") + 1]
                absolute_step = int(cmd[cmd.index("--steps") + 1])
                if _should_delete_resumed_checkpoint(variant, previous, shared_prefix_checkpoint):
                    previous.relative_to(args.output_dir.resolve())
                    previous.unlink(missing_ok=True)
    print(json.dumps({"command_count": len(commands), "executed": args.execute,
                      "data_dir": str(data_dir), "planned_steps": task_steps_summary(plan)},
                     ensure_ascii=False))


def task_steps_summary(plan: Mapping[str, Any]) -> dict[str, Any]:
    downstream = plan["downstream"]
    qa = qa_phase_steps(plan)
    return {"mlm_full_branch": int(plan["adaptation"]["full_branch_steps"]),
            "mlm_unique_executed": int(plan["adaptation"]["total_executed_steps"]),
            "classification": int(downstream["classification"]["steps"]),
            "qa_regular": qa["regular"], "qa_long": qa["long"],
            "retrieval": int(downstream["retrieval"]["steps"]),
            "reranking": int(downstream["reranking"]["steps"])}


if __name__ == "__main__":
    main()
