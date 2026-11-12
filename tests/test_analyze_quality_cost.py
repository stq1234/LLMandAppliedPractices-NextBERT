import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import analyze_quality_cost as analysis


class QualityCostAnalysisTests(unittest.TestCase):
    def test_macro_f1_uses_all_three_classes(self):
        self.assertAlmostEqual(analysis.macro_f1([0, 1, 2], [0, 1, 1]), 5 / 9)

    def test_qa_bootstrap_is_paired_and_preserves_question_weighted_point(self):
        left = {
            "q1": {"id": "q1", "language": "en", "f1": 0.0},
            "q2": {"id": "q2", "language": "en", "f1": 1.0},
            "q3": {"id": "q3", "language": "zh", "f1": 0.0},
        }
        right = {
            "q1": {"id": "q1", "language": "en", "f1": 1.0},
            "q2": {"id": "q2", "language": "en", "f1": 1.0},
            "q3": {"id": "q3", "language": "zh", "f1": 1.0},
        }
        clusters = {"q1": "en:a", "q2": "en:a", "q3": "zh:b"}
        points, deltas = analysis.paired_bootstrap("qa", {"V0": left, "V1": right}, ("f1",), analysis._qa_metric,
            reps=200, seed=7, clusters=clusters, point_unit="source_article_cluster", scale=100,
            comparisons=(("V0", "V1"),))
        by_variant = {(r["variant"], r["language"]): r for r in points}
        self.assertEqual(by_variant[("V0", "en")]["estimate"], 50.0)
        self.assertEqual(by_variant[("V1", "en")]["estimate"], 100.0)
        self.assertEqual(by_variant[("V1", "en_zh_macro")]["estimate"], 100.0)
        macro_delta = next(r for r in deltas if r["language"] == "en_zh_macro")
        self.assertEqual(macro_delta["delta"], 75.0)
        self.assertEqual(macro_delta["resampling_unit"], "source_article_cluster")

    def test_synchronized_language_bootstrap_uses_same_premise_draws(self):
        ids = ["en0", "en1", "zh0", "zh1"]
        languages = {"en0": "en", "en1": "en", "zh0": "zh", "zh1": "zh"}
        clusters = {"en0": "premise-a", "zh0": "premise-a",
                    "en1": "premise-b", "zh1": "premise-b"}
        samples = analysis._bootstrap_units(ids, languages, clusters, np.random.default_rng(5), 50, sync_languages=True)
        for en, zh in zip(samples["en"], samples["zh"]):
            en_groups = [clusters[ids[i]] for i in en]
            zh_groups = [clusters[ids[i]] for i in zh]
            self.assertEqual(en_groups, zh_groups)

    def test_xnli_clusters_use_english_premise_with_aligned_ids(self):
        rows = [
            {"id": "en:en:test:0", "language": "en", "premise": "Same premise", "label": 0},
            {"id": "en:en:test:1", "language": "en", "premise": "Same premise", "label": 1},
            {"id": "zh:zh:test:0", "language": "zh", "premise": "相同前提", "label": 0},
            {"id": "zh:zh:test:1", "language": "zh", "premise": "相同前提", "label": 1},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "classification_test.jsonl"
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            clusters, info = analysis._load_classification_clusters(path)
        self.assertEqual(clusters["en:en:test:0"], clusters["zh:zh:test:0"])
        self.assertEqual(clusters["en:en:test:1"], clusters["zh:zh:test:1"])
        self.assertEqual(clusters["en:en:test:0"], clusters["en:en:test:1"])
        self.assertEqual(info["label_mismatches"], 0)
        self.assertEqual(info["english_premise_clusters"], 1)

    def test_bm25_reconstruction_matches_top_candidate(self):
        row = {"id": "q-en-0", "language": "en", "labels": [1] + [0] * 19,
               "bm25_scores": list(range(20, 0, -1)),
               "candidate_docids": ["d1"] + [f"d{i}" for i in range(2, 21)],
               "relevant_doc_ids": ["d1"],
               "candidate_source": "fixed BM25 top-20 over retrieval_corpus.jsonl"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reranking_test.jsonl"
            path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            result = analysis._bm25_per_query(path)["q-en-0"]
        self.assertEqual(result["nDCG@10"], 1.0)
        self.assertEqual(result["MRR@10"], 1.0)
        self.assertEqual(result["candidate_Recall@20"], 1.0)

    def test_cost_parser_reads_nested_three_round_cells(self):
        cells = []
        for variant in ("V0", "V1", "V2", "V3"):
            for length, batch in ((512, 16), (4096, 4), (8192, 2)):
                rounds = []
                for index in range(1, 4):
                    rounds.append({"round": index, "summary": {
                        "forward_count": 12, "cuda_event_encode_ms_p50": 10.0,
                        "cuda_event_encode_ms_p95": 12.0,
                        "wall_prepared_cpu_gpu_mask_encode_ms_p50": 11.0,
                        "wall_prepared_cpu_gpu_mask_encode_ms_p95": 13.0,
                        "nonpad_tokens_per_second": 1000.0 + index},
                        "peak_memory": {"max_allocated_bytes": 100, "max_reserved_bytes": 200},
                        "samples": [{}] * 12})
                cells.append({"variant": variant, "sequence_length": length,
                              "physical_batch": batch, "rounds": rounds})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cost.json"
            path.write_text(json.dumps({"status": "completed", "cells": cells}), encoding="utf-8")
            normalized, _ = analysis.load_cost([path])
        row = next(r for r in normalized if r["variant"] == "V3" and r["length"] == 8192)
        self.assertEqual(row["rounds"], 3)
        self.assertEqual(row["tokens_per_second"], 1002.0)
        self.assertEqual(row["wall_latency_p50_s"], 0.011)
        self.assertEqual(row["peak_reserved_bytes"], 200)

    def test_long_context_equal_position_bootstrap_keeps_paired_groups(self):
        values = {}
        for vi, variant in enumerate(("V0", "V1", "V2", "V3")):
            values[variant] = {}
            for language in ("en", "zh"):
                values[variant][language] = {}
                for position in ("first", "middle", "last"):
                    values[variant][language][position] = {
                        f"{language}-g1": 0.1 * vi + 0.2,
                        f"{language}-g2": 0.1 * vi + 0.4,
                    }
        points, deltas, balanced, counts, _ = analysis._long_condition_analysis(
            values, task="long", metric="F1_equal_position", transform=lambda x: x,
            reps=200, rng=np.random.default_rng(11))
        self.assertEqual(counts["en"]["middle"], 2)
        self.assertEqual(counts["en"]["union"], 2)
        self.assertEqual(counts["en"]["complete_case"], 2)
        self.assertAlmostEqual(balanced["V3"] - balanced["V0"], 0.3)
        macro = next(r for r in points if r["variant"] == "V0" and r["language"] == "en_zh_macro")
        self.assertIsNotNone(macro["ci_low"])
        self.assertGreater(macro["ci_high"] - macro["ci_low"], 0.05)
        macro_delta = next(r for r in deltas if r["comparison"] == "V3-V0" and r["language"] == "en_zh_macro")
        self.assertAlmostEqual(macro_delta["delta"], 0.3)
        self.assertLessEqual(macro_delta["ci_low"], 0.3)
        self.assertGreaterEqual(macro_delta["ci_high"], 0.3)

    def test_long_point_uses_available_groups_per_position(self):
        position_values = {
            "first": {"g1": 0.1, "g2": 0.3},
            "middle": {"g1": 0.3, "g2": 0.5},
            "last": {"g2": 0.7, "g3": 0.9},
        }
        values = {}
        for vi, variant in enumerate(("V0", "V1", "V2", "V3")):
            values[variant] = {
                language: {
                    position: {group: score + vi * 0.03 for group, score in groups.items()}
                    for position, groups in position_values.items()
                }
                for language in ("en", "zh")
            }
        points, deltas, balanced, counts, complete_case = analysis._long_condition_analysis(
            values, task="long", metric="F1_equal_position", transform=lambda x: x,
            reps=200, rng=np.random.default_rng(13))
        self.assertAlmostEqual(balanced["V0"], (0.2 + 0.4 + 0.8) / 3)
        self.assertAlmostEqual(complete_case["V0"], (0.3 + 0.5 + 0.7) / 3)
        self.assertNotAlmostEqual(balanced["V0"], complete_case["V0"])
        self.assertEqual(counts["zh"], {"first": 2, "middle": 2, "last": 2, "union": 3, "complete_case": 1})
        self.assertAlmostEqual(balanced["V3"] - balanced["V0"], 0.09)
        macro_delta = next(r for r in deltas if r["comparison"] == "V3-V0" and r["language"] == "en_zh_macro")
        self.assertAlmostEqual(macro_delta["delta"], 0.09)

    def test_long_breakdown_keeps_views_and_deleted_answer_repetition(self):
        rows = [
            {"language": "en", "length": 8192, "evidence_position": "last", "condition": "intact",
             "view": "sliding512", "source_group_id": "g1", "exact_match": 1.0, "f1": 0.75},
            {"language": "en", "length": 8192, "evidence_position": "last", "condition": "evidence_deleted",
             "view": "full", "source_group_id": "g1", "exact_match": None, "f1": None, "fact_change_correct": 0},
            {"language": "en", "length": 4096, "evidence_position": "middle", "condition": "evidence_deleted",
             "view": "first512", "source_group_id": "g2", "exact_match": None, "f1": None, "fact_change_correct": 1},
        ]
        grouped = analysis._long_breakdown({"V0": rows})
        deleted = next(r for r in grouped if r["condition"] == "evidence_deleted" and r["view"] == "full")
        self.assertEqual(deleted["old_answer_reproduction_rate"], 1.0)
        self.assertIsNone(deleted["f1"])
        self.assertTrue(any(r["length"] == 4096 and r["view"] == "first512" for r in grouped))
        self.assertTrue(any(r["condition"] == "intact" and r["view"] == "sliding512" for r in grouped))


if __name__ == "__main__":
    unittest.main()
