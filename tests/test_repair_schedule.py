from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import retrain_gated_downstream as repair  # noqa: E402
from run_training_matrix import _load_plan, build_commands  # noqa: E402


class RepairScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.model = ROOT / "artifacts" / "mmbert-base"
        self.data = ROOT / "data" / "prepared" / "full_67m"
        self.original = self.root / "original"
        self.output = self.root / "repair"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_schedule_runs_exact_six_commands_on_fixed_cards_without_training_changes(self) -> None:
        jobs = repair.build_repair_schedule(self.model, self.data, self.original, self.output)
        expected_assignment = [
            (5, "V2", "retrieval"), (5, "V3", "retrieval"), (5, "V3", "qa"),
            (5, "V3", "classification"), (7, "V2", "qa"), (7, "V3", "reranking"),
        ]
        self.assertEqual(len(jobs), 6)
        self.assertEqual([(job["gpu"], job["variant"], job["task"]) for job in jobs], expected_assignment)
        self.assertEqual({job["gpu"] for job in jobs}, {5, 7})

        original_commands = build_commands(self.model, self.data, self.original, _load_plan())
        expected_by_key = {(command[command.index("--variant") + 1], command[command.index("--task") + 1]): command
                           for command in original_commands if command[command.index("--task") + 1] != "mlm"}
        for job in jobs:
            train_command = list(job["training_command"])
            expected = list(expected_by_key[(job["variant"], job["task"])])
            original_out = Path(expected[expected.index("--output-dir") + 1]).resolve()
            expected[expected.index("--output-dir") + 1] = str(
                self.output.resolve() / original_out.relative_to(self.original.resolve()))
            self.assertEqual(train_command, expected)
            init_from = Path(train_command[train_command.index("--init-from") + 1]).resolve()
            self.assertTrue(init_from.is_relative_to(self.original.resolve()))
            self.assertIn("mlm", init_from.parts)
            self.assertIn("--final-weights-only", train_command)
            self.assertEqual(job["steps"], int(train_command[train_command.index("--steps") + 1]))
            self.assertEqual(job["command"], [
                sys.executable, str(repair.WORKER), "--gpu", str(job["gpu"]), "--", *train_command[2:],
            ])

    def test_memory_preflight_queries_only_assigned_cards_and_waits_for_both(self) -> None:
        observations = {
            5: [{"free_gib": 38.0, "total_gib": 48.0}, {"free_gib": 45.0, "total_gib": 48.0}],
            7: [{"free_gib": 26.0, "total_gib": 48.0}, {"free_gib": 26.0, "total_gib": 48.0}],
        }
        seen: list[int] = []

        def query(gpu: int) -> dict[str, float]:
            seen.append(gpu)
            return observations[gpu][0 if seen.count(gpu) == 1 else 1]

        with patch.object(repair, "query_gpu", side_effect=query), patch.object(repair.time, "sleep") as sleep:
            ready = repair.wait_for_gpus(poll_seconds=30)
        self.assertEqual(seen, [5, 7, 5, 7])
        sleep.assert_called_once_with(30)
        self.assertEqual(ready["5"]["free_gib"], 45.0)

    def _fake_jobs(self) -> list[dict[str, object]]:
        jobs = []
        assignments = [*repair.GPU_TASKS[5], *repair.GPU_TASKS[7]]
        gpu_for = {pair: gpu for gpu, pairs in repair.GPU_TASKS.items() for pair in pairs}
        for variant, task in assignments:
            gpu = gpu_for[(variant, task)]
            job_id = f"{variant}_{task}_seed17"
            directory = self.output / "downstream" / variant / task / "17"
            data, initial = self.root / f"{job_id}.jsonl", self.root / f"{job_id}_init.pt"
            data.write_text("{}\n", encoding="utf-8")
            initial.write_bytes(b"init")
            training = ["python", "train_nextbert.py", "--task", task, "--variant", variant,
                        "--output-dir", str(directory)]
            jobs.append({
                "job_id": job_id, "gpu": gpu, "variant": variant, "task": task,
                "seed": 17, "steps": 2, "training_command": training,
                "command": ["fake-worker", "--gpu", str(gpu), "--", *training[2:]],
                "data": str(data), "init_checkpoint": str(initial), "output_dir": str(directory),
                "checkpoint": str(directory / f"{variant}_{task}_checkpoint.pt"),
                "step_log": str(directory / f"{variant}_{task}_training.jsonl"),
            })
        return jobs

    def test_fake_workers_verify_concurrency_outputs_and_gpu_summaries(self) -> None:
        jobs = self._fake_jobs()
        active = 0
        peak_active = 0
        lock = threading.Lock()
        barrier = threading.Barrier(6)
        job_by_key = {(job["variant"], job["task"]): job for job in jobs}

        class FakeProcess:
            next_pid = 62000

            def __init__(self, command: list[str], **kwargs: object) -> None:
                nonlocal active, peak_active
                self.gpu = int(command[command.index("--gpu") + 1])
                args = command[command.index("--") + 1:]
                self.variant = args[args.index("--variant") + 1]
                self.task = args[args.index("--task") + 1]
                self.job = job_by_key[(self.variant, self.task)]
                self.stdout = kwargs["stdout"]
                self.returncode: int | None = None
                FakeProcess.next_pid += 1
                self.pid = FakeProcess.next_pid
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)

            def wait(self, timeout: float | None = None) -> int:
                nonlocal active
                del timeout
                self.stdout.write("__WORKER_START__=" + json.dumps({
                    "pid": self.pid, "physical_gpu": self.gpu,
                    "logical_gpu": 0, "device_name": f"Fake GPU {self.gpu}",
                    "started_unix": time.time(),
                }) + "\n")
                barrier.wait(timeout=5)
                directory = Path(str(self.job["output_dir"]))
                directory.mkdir(parents=True, exist_ok=True)
                Path(str(self.job["checkpoint"])).write_bytes(b"fake checkpoint")
                Path(str(self.job["step_log"])).write_text("".join(
                    json.dumps({"step": step, "nonpad_tokens": 100,
                                "optimizer_step_seconds": 0.5,
                                "memory": {"peak_allocated_bytes": self.gpu * 100,
                                           "peak_reserved_bytes": self.gpu * 200}}) + "\n"
                    for step in (1, 2)), encoding="utf-8")
                self.stdout.write("__WORKER_END__=" + json.dumps({
                    "pid": self.pid, "success": True,
                    "after_runpy": {"peak_allocated_bytes": self.gpu * 100,
                                    "peak_reserved_bytes": self.gpu * 200},
                }) + "\n")
                self.returncode = 0
                with lock:
                    active -= 1
                return 0

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.terminate()

        self.output.mkdir(parents=True)
        with patch.object(repair.subprocess, "Popen", FakeProcess):
            status = repair._execute_jobs(
                jobs, output=self.output,
                initial_gpus={"5": {"free_gib": 47.0, "total_gib": 48.0},
                              "7": {"free_gib": 26.0, "total_gib": 48.0}},
            )
        self.assertEqual(peak_active, 6)
        self.assertEqual(status["status"], "completed")
        self.assertTrue(all(Path(job["checkpoint"]).is_file() for job in jobs))
        gpu5, gpu7 = status["per_gpu"]["5"], status["per_gpu"]["7"]
        self.assertEqual((gpu5["job_count"], gpu7["job_count"]), (4, 2))
        self.assertEqual((gpu5["optimizer_updates"], gpu7["optimizer_updates"]), (8, 4))
        self.assertEqual((gpu5["nonpad_tokens"], gpu7["nonpad_tokens"]), (800, 400))
        self.assertEqual((gpu5["peak_allocated_bytes"], gpu7["peak_allocated_bytes"]), (500, 700))
        self.assertLessEqual(gpu5["wall_clock_union_seconds"], gpu5["worker_runtime_seconds_sum"] + 0.05)
        saved = json.loads((self.output / "repair_status.json").read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "completed")
        self.assertEqual(saved["run_id"], "gatefix_57")

    def test_fake_worker_failure_sets_failed_status_and_propagates(self) -> None:
        jobs = self._fake_jobs()
        lock = threading.Lock()
        barrier = threading.Barrier(6)
        terminated = threading.Event()
        job_by_key = {(job["variant"], job["task"]): job for job in jobs}

        class FakeProcess:
            next_pid = 63000

            def __init__(self, command: list[str], **kwargs: object) -> None:
                del kwargs
                self.gpu = int(command[command.index("--gpu") + 1])
                args = command[command.index("--") + 1:]
                self.variant = args[args.index("--variant") + 1]
                self.task = args[args.index("--task") + 1]
                self.job_id = job_by_key[(self.variant, self.task)]["job_id"]
                self.returncode: int | None = None
                FakeProcess.next_pid += 1
                self.pid = FakeProcess.next_pid

            def wait(self, timeout: float | None = None) -> int:
                del timeout
                barrier.wait(timeout=5)
                if self.job_id == jobs[0]["job_id"]:
                    self.returncode = 7
                    return 7
                terminated.wait(timeout=5)
                return self.returncode if self.returncode is not None else 0

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15
                terminated.set()

            def kill(self) -> None:
                self.terminate()

        self.output.mkdir(parents=True)
        with patch.object(repair.subprocess, "Popen", FakeProcess):
            status = repair._execute_jobs(
                jobs, output=self.output,
                initial_gpus={"5": {"free_gib": 47.0, "total_gib": 48.0},
                              "7": {"free_gib": 26.0, "total_gib": 48.0}},
            )
        self.assertEqual(status["status"], "failed")
        self.assertIsNotNone(status["failure"])
        self.assertEqual(json.loads((self.output / "repair_status.json").read_text(encoding="utf-8"))["status"], "failed")


if __name__ == "__main__":
    unittest.main()
