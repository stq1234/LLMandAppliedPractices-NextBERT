from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_all_evaluation as evaluation  # noqa: E402


class EvaluationScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.model = self.base / "model"
        self.data = self.base / "data"
        self.training = self.base / "training"
        self.gated = self.base / "gatefix"
        self.output = self.base / "evaluation"

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fixed_variant_matrix_and_gatefix_checkpoint_selection(self) -> None:
        jobs = evaluation.build_task_jobs(
            model_dir=self.model, data_dir=self.data, training_dir=self.training,
            gated_training_dir=self.gated, output_dir=self.output,
            variants=evaluation.select_variants(),
        )
        self.assertEqual(len(jobs), 20)
        self.assertEqual(sum(job["task"] == "retrieval" for job in jobs), 4)
        self.assertEqual(sum(job["task"] != "retrieval" for job in jobs), 16)
        self.assertEqual([job["variant"] for job in jobs if job["lane"] == "A"][:7], ["V0"] * 7)
        self.assertEqual([job["variant"] for job in jobs if job["lane"] == "B"][:7], ["V3"] * 7)
        for job in jobs:
            checkpoint = Path(job["checkpoint"])
            expected_root = self.gated if job["variant"] in {"V2", "V3"} else self.training
            self.assertEqual(checkpoint.parts[:len(expected_root.parts)], expected_root.parts)
            self.assertIn("--device", job["command"])
            self.assertNotIn("--sample-queries", job["command"])
            self.assertNotIn("--sample-documents-per-language", job["command"])
            if job["task"] == "reranking":
                self.assertIn("--require-20-candidates", job["command"])

        selected = evaluation.build_task_jobs(
            model_dir=self.model, data_dir=self.data, training_dir=self.training,
            gated_training_dir=self.gated, output_dir=self.output,
            variants=evaluation.select_variants(exclude=("V2", "V3")),
        )
        self.assertEqual(len(selected), 10)
        self.assertTrue(all(job["variant"] in {"V0", "V1"} for job in selected))

    def test_retrieval_queries_are_combined_once_and_metrics_split_by_source(self) -> None:
        dev = self.base / "dev.jsonl"
        test = self.base / "test.jsonl"
        combined = self.output / "inputs" / "combined.jsonl"
        dev_rows = [
            {"id": "same", "language": "en", "query_ids": [1]},
            {"id": "same", "language": "zh", "query_ids": [2]},
        ]
        test_rows = [{"id": "test-1", "language": "en", "query_ids": [3]}]
        dev.write_text("".join(json.dumps(row) + "\n" for row in dev_rows), encoding="utf-8")
        test.write_text("".join(json.dumps(row) + "\n" for row in test_rows), encoding="utf-8")

        source_map, counts = evaluation.combine_retrieval_queries(dev, test, combined)
        self.assertEqual(counts, {"dev": 2, "test": 1})
        self.assertEqual(source_map[("en", "same")], "dev")
        self.assertEqual(source_map[("zh", "same")], "dev")
        self.assertEqual(source_map[("en", "test-1")], "test")
        self.assertEqual(len(combined.read_text(encoding="utf-8").splitlines()), 3)

        predictions = {
            "en": [
                {"id": "same", "language": "en", "metrics": {"nDCG@10": 1.0, "Recall@10": 1.0, "Recall@100": 1.0}},
                {"id": "test-1", "language": "en", "metrics": {"nDCG@10": 0.5, "Recall@10": 0.0, "Recall@100": 1.0}},
            ],
            "zh": [
                {"id": "same", "language": "zh", "metrics": {"nDCG@10": 0.0, "Recall@10": 0.0, "Recall@100": 0.0}},
            ],
        }
        source_result = {"languages": [
            {"language": language, "documents_indexed": 10, "query_predictions": rows}
            for language, rows in predictions.items()
        ]}
        result = evaluation.aggregate_retrieval_result(source_result, source_map, counts, variant="V0")
        self.assertEqual(result["queries_evaluated"], 3)
        self.assertEqual(result["by_split"]["dev"]["queries"], 2)
        self.assertEqual(result["by_split"]["test"]["queries"], 1)
        self.assertEqual(result["by_language"]["en"]["dev"]["queries"], 1)
        self.assertEqual(result["by_language"]["en"]["test"]["queries"], 1)
        self.assertEqual(result["by_language"]["zh"]["test"]["queries"], 0)
        self.assertEqual(result["by_language"]["en"]["test"]["Recall@10"], 0.0)

    def test_duplicate_retrieval_ids_across_splits_are_rejected(self) -> None:
        dev = self.base / "dev.jsonl"
        test = self.base / "test.jsonl"
        dev.write_text('{"id":"q1","language":"en"}\n', encoding="utf-8")
        test.write_text('{"id":"q1","language":"en"}\n', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate retrieval"):
            evaluation.combine_retrieval_queries(dev, test, self.output / "combined.jsonl")

    def test_diagnostic_matrix_uses_fixed_protocols_and_gatefix_retrieval_states(self) -> None:
        jobs = evaluation.build_diagnostic_jobs(
            model_dir=self.model, data_dir=self.data, training_dir=self.training,
            gated_training_dir=self.gated, output_dir=self.output,
            variants=evaluation.select_variants(),
        )
        self.assertEqual(len(jobs), 35)
        self.assertEqual(sum(job["task"] == "long" for job in jobs), 4)
        self.assertEqual(sum(job["task"] == "performance" for job in jobs), 12)
        self.assertEqual(sum(job["task"] == "mlm_retention" for job in jobs), 14)
        self.assertEqual(sum(job["task"] == "probe" for job in jobs), 5)
        self.assertTrue(all(job["lane"] == "performance" for job in jobs if job["task"] == "performance"))

        long_jobs = [job for job in jobs if job["task"] == "long"]
        for job in long_jobs:
            command = job["command"]
            self.assertEqual(command[command.index("--profile") + 1], "full_67m")
            self.assertEqual(command[command.index("--max-source-groups") + 1], "50")
            self.assertEqual(command[command.index("--eval-batch-size") + 1], "4")

        retention = {job["job_id"]: job for job in jobs if job["task"] == "mlm_retention"}
        baseline = retention["diagnostics_mlm_retention_original_V0_512"]
        self.assertIsNone(baseline["checkpoint"])
        v3_mlm = retention["diagnostics_mlm_retention_mlm_V3_512"]
        v3_retrieval = retention["diagnostics_mlm_retention_retrieval_V3_512"]
        self.assertEqual(Path(v3_mlm["checkpoint"]).parts[:len(self.training.parts)], self.training.parts)
        self.assertEqual(Path(v3_retrieval["checkpoint"]).parts[:len(self.gated.parts)], self.gated.parts)
        self.assertEqual(v3_retrieval["command"][v3_retrieval["command"].index("--seed") + 1], "1701")
        self.assertEqual(v3_retrieval["command"][v3_retrieval["command"].index("--batch-size") + 1], "4")

        probes = [job for job in jobs if job["task"] == "probe"]
        self.assertEqual({job["job_id"] for job in probes}, {
            "diagnostics_probe_original_V0", "diagnostics_probe_mlm_V0", "diagnostics_probe_mlm_V3",
            "diagnostics_probe_retrieval_V0", "diagnostics_probe_retrieval_V3",
        })
        self.assertTrue(all(job["command"][job["command"].index("--steps") + 1] == "100"
                            for job in probes))

    def test_repair_gate_wait_accepts_completed_machine_status(self) -> None:
        status = self.base / "repair_status.json"
        status.write_text('{"status":"completed"}\n', encoding="utf-8")
        evaluation._wait_for_repair_status(status)

    def test_repair_gate_rejects_failed_machine_status(self) -> None:
        status = self.base / "repair_status.json"
        status.write_text('{"status":"failed"}\n', encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "did not complete successfully"):
            evaluation._wait_for_repair_status(status)

    def test_performance_jobs_start_after_both_lanes_and_run_serially(self) -> None:
        jobs = []
        for lane in ("A", "B"):
            job_id = f"ordinary_{lane}"
            output = self.output / f"{job_id}.json"
            jobs.append({"job_id": job_id, "lane": lane, "task": "qa", "variant": "V0",
                         "command": [job_id, "qa", "V0", str(output)], "output": output})
        for index in range(3):
            job_id = f"performance_{index}"
            output = self.output / f"{job_id}.json"
            jobs.append({"job_id": job_id, "lane": "A", "task": "performance", "variant": "V0",
                         "command": [job_id, "performance", "V0", str(output)], "output": output})

        self.output.mkdir(parents=True)
        state = {"jobs": {}, "status": "running"}
        lock = threading.Lock()
        state_check_failures: list[str] = []
        performance_active = 0
        max_performance_active = 0

        class FakeProcess:
            _pid = 50000

            def __init__(self, command: list[str], **_kwargs: object) -> None:
                nonlocal performance_active, max_performance_active
                self.job_id = command[0]
                self.task = command[1]
                self.variant = command[2]
                self.output = Path(command[3])
                self.returncode: int | None = None
                FakeProcess._pid += 1
                self.pid = FakeProcess._pid
                if self.task == "performance":
                    with lock:
                        unfinished = [job["job_id"] for job in jobs if job["task"] != "performance"
                                      and state["jobs"].get(job["job_id"], {}).get("status") != "succeeded"]
                        state_check_failures.extend(unfinished)
                        performance_active += 1
                        max_performance_active = max(max_performance_active, performance_active)

            def wait(self, timeout: float | None = None) -> int:
                nonlocal performance_active
                del timeout
                self.output.parent.mkdir(parents=True, exist_ok=True)
                self.output.write_text(json.dumps({"task": self.task, "variant": self.variant}), encoding="utf-8")
                self.returncode = 0
                if self.task == "performance":
                    with lock:
                        performance_active -= 1
                return 0

            def poll(self) -> int | None:
                return self.returncode

            def terminate(self) -> None:
                self.returncode = -15

            def kill(self) -> None:
                self.terminate()

        with patch.object(evaluation.subprocess, "Popen", FakeProcess):
            completed = evaluation._run_lanes(
                jobs, environment={}, output_dir=self.output, model_dir=self.model,
                split_maps={}, state=state, status_path=self.output / evaluation.STATUS_NAME,
                provenance={job["job_id"]: {"test": job["job_id"]} for job in jobs},
                skipped=set(), max_workers=2,
            )
        self.assertTrue(completed)
        self.assertEqual(state["status"], "completed")
        self.assertEqual(state_check_failures, [])
        self.assertEqual(max_performance_active, 1)
        self.assertTrue(all(entry["status"] == "succeeded" for entry in state["jobs"].values()))


if __name__ == "__main__":
    unittest.main()
