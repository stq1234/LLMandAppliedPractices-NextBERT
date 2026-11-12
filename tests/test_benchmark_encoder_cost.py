from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import benchmark_encoder_cost as benchmark  # noqa: E402


class EncoderCostBenchmarkTests(unittest.TestCase):
    def test_plan_covers_every_cell_three_times_with_rotated_model_order(self) -> None:
        plan = benchmark.build_plan()
        self.assertEqual(len(plan), 36)
        self.assertEqual(benchmark.rotated_orders(), [
            ["V0", "V1", "V2", "V3"],
            ["V1", "V2", "V3", "V0"],
            ["V2", "V3", "V0", "V1"],
        ])
        counts = {}
        for cell in plan:
            key = (cell["variant"], cell["length"])
            counts[key] = counts.get(key, 0) + 1
            self.assertEqual(cell["physical_batch"], benchmark.BATCH_SIZES[cell["length"]])
        self.assertEqual(set(counts.values()), {3})
        self.assertEqual(set(counts), {(variant, length) for variant in benchmark.VARIANTS
                                       for length in benchmark.LENGTHS})

    def test_paths_and_fixed_heldout_batch_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                benchmark.checkpoint_path(root, "V3"),
                root / "mlm" / "V3" / "8192" / "V3_mlm_checkpoint.pt",
            )
            path = root / "wiki_512_test.npy"
            data = np.zeros((20, 512), dtype=np.int32)
            data[:, :7] = 9
            np.save(path, data)
            batch, meta = benchmark.load_prepared_batch(path, 512, 16)
            self.assertEqual(batch.shape, (16, 512))
            self.assertEqual(meta["selected_row_indices"], list(range(16)))
            self.assertEqual(meta["nonpad_tokens_per_batch"], 16 * 7)
            self.assertEqual(len(meta["sha256"]), 64)
            self.assertEqual(len(meta["selected_input_sha256"]), 64)

    def test_invalid_gate_checkpoint_is_rejected(self) -> None:
        benchmark.validate_gate_state("V0", {"gate_ramp": 0.0})
        benchmark.validate_gate_state("V2", {"gate_ramp": 1.0})
        with self.assertRaisesRegex(ValueError, "fully active"):
            benchmark.validate_gate_state("V3", {"gate_ramp": 0.0})

    def test_summary_uses_total_tokens_over_total_wall_time_as_primary_throughput(self) -> None:
        result = benchmark.summarize_samples([
            {"cuda_event_encode_ms": 10.0,
             "wall_prepared_cpu_gpu_mask_encode_ms": 1000.0,
             "nonpad_tokens": 100, "nonpad_tokens_per_second": 100.0},
            {"cuda_event_encode_ms": 5.0,
             "wall_prepared_cpu_gpu_mask_encode_ms": 100.0,
             "nonpad_tokens": 100, "nonpad_tokens_per_second": 1000.0},
        ])
        self.assertAlmostEqual(result["nonpad_tokens_per_second"], 200 / 1.1)
        self.assertEqual(result["nonpad_tokens_per_second_sample_mean"], 550.0)


if __name__ == "__main__":
    unittest.main()
