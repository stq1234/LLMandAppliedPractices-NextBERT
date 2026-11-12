import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from evaluate_nextbert import _metric_by_language
from correct_qa_language_metrics import correct_result


def test_partial_f1_is_averaged_numerically_instead_of_counting_nonzero_answers():
    result = _metric_by_language([
        {"language": "zh", "f1": 0.25}, {"language": "zh", "f1": 0.75},
        {"language": "en", "f1": 0.0},
    ], "f1")
    assert result["zh"] == {"f1": 0.5, "count": 2}
    assert result["en"] == {"f1": 0.0, "count": 1}


def test_cpu_correction_preserves_predictions_and_an_exact_original_snapshot(tmp_path):
    path = tmp_path / "qa.json"
    predictions = [{"language": "zh", "f1": 0.25, "exact_match": 0.0},
                   {"language": "en", "f1": 0.75, "exact_match": 0.0}]
    payload = {"task": "qa", "evaluation_scope": "prepared_data_scope", "question_count": 2,
               "f1": 0.5, "exact_match": 0.0, "predictions": predictions,
               "f1_by_language": {"zh": {"f1": 1.0, "count": 1}, "en": {"f1": 1.0, "count": 1}}}
    original = json.dumps(payload).encode()
    path.write_bytes(original)
    assert correct_result(path)["changed"]
    assert path.read_bytes() == original
    correct_result(path, execute=True)
    corrected = json.loads(path.read_bytes())
    assert corrected["predictions"] == predictions
    assert corrected["f1_by_language"]["zh"]["f1"] == 0.25
    assert path.with_name("qa.before_language_f1_fix.json").read_bytes() == original
    assert not correct_result(path, execute=True)["changed"]
