"""Check the Python 3 metrics against the upstream CMRC evaluator."""
import importlib.util
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from qa_metrics import answer_em, answer_f1


def test_squad_normalization_and_f1():
    assert answer_em('The blue-car.', 'bluecar', 'en') == 1
    assert answer_f1('red blue', 'blue green', 'en') == 0.5


def test_cmrc_uses_contiguous_match():
    assert answer_f1('甲乙丙', '甲丁丙', 'zh') == pytest.approx(1 / 3)
    assert answer_em('北京 上海', '北京上海', 'zh') == 0


def test_cmrc_matches_official_python3_reference():
    source = Path(__file__).parent / 'fixtures' / 'cmrc2018_official.py'
    spec = importlib.util.spec_from_file_location('cmrc_official', source)
    official = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(official)
    pairs = [('甲乙丙', '甲丁丙'), ('北京，上海。', '北京上海'),
             ('OpenAI GPT-4', 'OpenAI GPT4'), ('1990年', '1991年'),
             ('The U.S.A.', 'U.S.A.'), ('北京 上海', '北京上海'), ('', '')]
    for prediction, gold in pairs:
        assert answer_f1(prediction, gold, 'zh') == pytest.approx(
            official.calc_f1_score([gold], prediction))
        assert answer_em(prediction, gold, 'zh') == official.calc_em_score([gold], prediction)
