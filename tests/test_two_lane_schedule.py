from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import run_two_lane_training as scheduler  # noqa: E402
from run_training_matrix import _load_plan, build_commands  # noqa: E402


def _arg(command: list[str], name: str) -> str | None:
    try:
        return command[command.index(name) + 1]
    except (ValueError, IndexError):
        return None


class TwoLaneScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "formal"
        self.plan = _load_plan()
        commands = build_commands(
            ROOT / "artifacts/mmbert-base",
            ROOT / "data/prepared/full_67m",
            self.output,
            self.plan,
        )
        self.commands = commands
        self.waves = scheduler.build_schedule(commands, self.plan)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_fixed_five_waves_cover_fourteen_mlm_and_twelve_downstream_once(self) -> None:
        self.assertEqual(len(self.waves), 5)
        scheduled = [job for wave in self.waves for lane in wave["lanes"].values()
                     for job in lane]
        self.assertEqual(len(scheduled), 26)
        self.assertEqual(len({job["index"] for job in scheduled}), 26)
        self.assertEqual(len({job["job_id"] for job in scheduled}), 26)
        self.assertEqual(sum(job["task"] == "mlm" for job in scheduled), 14)
        downstream = [job for job in scheduled if job["task"] != "mlm"]
        self.assertEqual(len(downstream), 12)
        self.assertEqual(len({(job["variant"], job["task"]) for job in downstream}), 12)
        self.assertTrue(all(set(wave["lanes"]) == {"A", "B"} for wave in self.waves))
        self.assertLessEqual(max(len(wave["lanes"]) for wave in self.waves), 2)

    def test_lane_order_matches_the_shared_prefix_and_downstream_dependencies(self) -> None:
        task_sequences = [
            [[job["task"] for job in wave["lanes"][lane]] for lane in ("A", "B")]
            for wave in self.waves
        ]
        self.assertEqual(task_sequences[1][1], ["classification", "reranking", "qa"])
        self.assertEqual(task_sequences[3][0], ["qa", "retrieval"])
        self.assertEqual(task_sequences[3][1], ["classification", "reranking", "retrieval"])
        self.assertEqual(task_sequences[4][0], ["qa", "retrieval"])
        self.assertEqual(task_sequences[4][1], ["qa", "retrieval"])

        v2_shared = self.waves[1]["lanes"]["A"]
        v2_tail = self.waves[2]["lanes"]["A"]
        v3_tail = self.waves[2]["lanes"]["B"]
        self.assertEqual([job["steps"] for job in v2_shared], [192, 384])
        self.assertEqual([job["steps"] for job in v2_tail], [448, 512])
        self.assertEqual([job["steps"] for job in v3_tail], [448, 512])
        shared_checkpoint = Path(_arg(v2_shared[-1]["command"], "--output-dir")) / "V2_mlm_checkpoint.pt"
        for command_job in (v2_tail[0], v3_tail[0]):
            self.assertEqual(Path(_arg(command_job["command"], "--resume-from")), shared_checkpoint)
        self.assertIn("--shared-prefix-resume", v3_tail[0]["command"])

    def test_optimizer_state_and_allocator_budgets_are_preserved(self) -> None:
        jobs = [job for wave in self.waves for lane in wave["lanes"].values()
                for job in lane]
        mlm_jobs = [job for job in jobs if job["task"] == "mlm"]
        self.assertTrue(all("--schedule-total-steps" in job["command"] for job in mlm_jobs))
        self.assertTrue(all("--final-weights-only" not in job["command"] for job in mlm_jobs))
        downstream = [job for job in jobs if job["task"] != "mlm"]
        self.assertTrue(all("--final-weights-only" in job["command"] for job in downstream))
        self.assertTrue(all("--init-from" in job["command"] for job in downstream))

        budgets = {(job["task"], job["memory_budget_gib"], job["minimum_free_gib"])
                   for job in jobs}
        self.assertIn(("mlm", 12.0, 18.0), budgets)
        self.assertIn(("qa", 11.0, 17.0), budgets)
        self.assertIn(("classification", 8.0, 14.0), budgets)
        self.assertIn(("reranking", 8.0, 14.0), budgets)
        public = scheduler._public_plan(
            self.waves, gpu=7, max_workers=1, data_dir=ROOT / "data/prepared/full_67m",
            output_dir=self.output, model_dir=ROOT / "artifacts/mmbert-base",
        )
        self.assertEqual(public["maximum_concurrent_tasks"], 1)
        self.assertEqual(public["global_preflight_minimum_free_gib"], 18.0)
        self.assertEqual(public["downstream_job_count"], 12)
        parallel = scheduler._public_plan(
            self.waves, gpu=7, max_workers=2, data_dir=ROOT / "data/prepared/full_67m",
            output_dir=self.output, model_dir=ROOT / "artifacts/mmbert-base",
        )
        self.assertEqual(parallel["maximum_concurrent_tasks"], 2)
        self.assertEqual(parallel["global_preflight_minimum_free_gib"], 30.0)

    def test_execution_modes_obey_worker_limit_and_save_final_checkpoints(self) -> None:
        for max_workers in (1, 2):
            with self.subTest(max_workers=max_workers):
                output = Path(self.temp.name) / f"execution-{max_workers}"
                commands = build_commands(
                    ROOT / "artifacts/mmbert-base", ROOT / "data/prepared/full_67m",
                    output, self.plan,
                )
                waves = scheduler.build_schedule(commands, self.plan)
                state = {"active": 0, "maximum": 0, "next_pid": 50000}
                lock = threading.Lock()
                first_wave_barrier = threading.Barrier(2) if max_workers == 2 else None

                class FakeProcess:
                    def __init__(self, command: list[str], **_kwargs: object) -> None:
                        self.command = command
                        self.returncode: int | None = None
                        with lock:
                            state["active"] += 1
                            state["maximum"] = max(state["maximum"], state["active"])
                            state["next_pid"] += 1
                            self.pid = int(state["next_pid"])

                    def wait(self, timeout: float | None = None) -> int:
                        del timeout
                        training_args = self.command[self.command.index("--") + 1:]
                        task = _arg(training_args, "--task")
                        variant = _arg(training_args, "--variant")
                        output_dir = Path(_arg(training_args, "--output-dir"))
                        if (first_wave_barrier is not None and task == "mlm"
                                and variant in {"V0", "V1"}
                                and _arg(training_args, "--length") == "512"):
                            first_wave_barrier.wait(timeout=3)
                        checkpoint = output_dir / f"{variant}_{task}_checkpoint.pt"
                        checkpoint.parent.mkdir(parents=True, exist_ok=True)
                        checkpoint.write_bytes(b"checkpoint")
                        self.returncode = 0
                        with lock:
                            state["active"] -= 1
                        return 0

                    def poll(self) -> int | None:
                        return self.returncode

                    def terminate(self) -> None:
                        self.returncode = -15

                    def kill(self) -> None:
                        self.terminate()

                with patch.object(scheduler.subprocess, "Popen", FakeProcess), redirect_stdout(StringIO()):
                    result = scheduler._execute_waves(
                        waves, gpu=7, output_dir=output, env={}, max_workers=max_workers,
                    )

                self.assertEqual(result["status"], "completed")
                self.assertEqual(state["maximum"], max_workers)
                self.assertEqual(len(result["removed_checkpoints"]), 10)
                for variant in ("V0", "V1", "V2", "V3"):
                    final_mlm = output / "mlm" / variant / "8192" / f"{variant}_mlm_checkpoint.pt"
                    self.assertTrue(final_mlm.is_file())
                downstream = list((output / "downstream").rglob("*_checkpoint.pt"))
                self.assertEqual(len(downstream), 12)

    def test_gpu_wait_retries_until_process_owner_and_memory_are_ready(self) -> None:
        gib = 1024**3
        process_rows = [
            [{"pid": 20, "owner": "another-user"}],
            [{"pid": 20, "owner": "shaotianqi"}, {"pid": 21, "owner": "another-user"}],
            [{"pid": 20, "owner": "shaotianqi"}],
        ]
        memory_rows = [(35 * gib, 48 * gib), (35 * gib, 48 * gib), (31 * gib, 48 * gib)]
        with patch.object(scheduler, "query_gpu_processes", side_effect=process_rows) as processes, \
                patch.object(scheduler, "query_gpu_memory", side_effect=memory_rows) as memory, \
                patch.object(scheduler.time, "sleep") as sleep:
            result = scheduler.wait_for_gpu(
                gpu=7, max_workers=2, poll_seconds=30, allowed_owner="shaotianqi",
            )

        self.assertEqual(result, (31 * gib, 48 * gib))
        self.assertEqual(processes.call_count, 3)
        self.assertEqual(memory.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(30)

    def test_gpu_wait_does_not_accept_other_owner_multiple_processes_or_low_memory(self) -> None:
        gib = 1024**3
        invalid_states = [
            ([{"pid": 20, "owner": "another-user"}], (35 * gib, 48 * gib)),
            ([{"pid": 20, "owner": "shaotianqi"}, {"pid": 21, "owner": "shaotianqi"}],
             (35 * gib, 48 * gib)),
            ([], (29 * gib, 48 * gib)),
        ]
        for process_rows, memory_row in invalid_states:
            with self.subTest(processes=process_rows, free=memory_row[0]):
                with patch.object(scheduler, "query_gpu_processes", return_value=process_rows), \
                        patch.object(scheduler, "query_gpu_memory", return_value=memory_row), \
                        patch.object(scheduler.time, "sleep", side_effect=KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt):
                        scheduler.wait_for_gpu(
                            gpu=7, max_workers=2, poll_seconds=30, allowed_owner="shaotianqi",
                        )

    def test_cli_dry_run_does_not_wait_for_gpu_or_execute(self) -> None:
        with patch.object(scheduler, "wait_for_gpu") as wait, \
                patch.object(scheduler, "_execute_waves") as execute, \
                redirect_stdout(StringIO()):
            result = scheduler.main([
                "--gpu", "7", "--output-dir", str(Path(self.temp.name) / "dry-run"),
            ])
        self.assertEqual(result, 0)
        wait.assert_not_called()
        execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
