import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import prepare_data


def test_corrects_a_unique_nearest_answer_offset_without_changing_text():
    text = "是答案"
    row = {
        "context": "x 是答案",
        "question": "是什么？",
        "answers": {"text": [text], "answer_start": [1]},
    }

    normalized, status, reason, answer_pairs, corrected_pairs = prepare_data.normalize_qa_answers(row)

    assert status == "corrected_nearest"
    assert reason == "nearest_exact_answer_occurrence"
    assert answer_pairs == corrected_pairs == 1
    assert normalized["answers"]["text"] == [text]
    assert normalized["answers"]["answer_start"] == [2]
    assert normalized["source_answers_original"]["answer_start"] == [1]
    assert normalized["context"][2:2 + len(text)] == text


def test_keeps_ambiguous_evaluation_gold_unaligned_for_denominator():
    row = {
        "context": "aba",
        "question": "哪一个？",
        "answers": {"text": ["a"], "answer_start": [1]},
    }

    normalized, status, reason, answer_pairs, corrected_pairs = prepare_data.normalize_qa_answers(row, preserve_unalignable=True)

    assert status == "unalignable_eval"
    assert reason == "ambiguous_nearest_answer_occurrence"
    assert answer_pairs == 1
    assert corrected_pairs == 0
    assert normalized["answers"] == row["answers"]
    assert normalized["answer_alignment_status"] == "unalignable_eval"


def test_plan_snapshot_preserves_the_exact_input_bytes(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    plan_path = project_root / "configs" / "experiment_plan.json"
    plan_path.parent.mkdir(parents=True)
    monkeypatch.setattr(prepare_data, "ROOT", project_root)
    monkeypatch.setattr(prepare_data, "RAW_ROOT", project_root / "artifacts" / "raw")

    plan_bytes = b'{"data_profile":"profile-a"}\r\n'
    plan_path.write_bytes(plan_bytes)
    plan_relative, snapshot_relative = prepare_data.save_plan_snapshot("profile-a", plan_path, plan_bytes)

    snapshot_path = project_root / snapshot_relative
    assert Path(plan_relative).as_posix() == "configs/experiment_plan.json"
    assert snapshot_path.read_bytes() == plan_bytes
    assert prepare_data.sha256_file(snapshot_path) == prepare_data.sha256_bytes(plan_bytes)
    assert Path(snapshot_relative).as_posix().startswith("artifacts/raw/profile-a/provenance/")
