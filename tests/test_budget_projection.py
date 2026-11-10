"""Exercise evaluation projection against the maintained plan schema."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from run_pilot import read_plan, _evaluation_cost_projection


def test_real_plan_covers_dev_and_test_reranking_budget():
    plan = read_plan()
    cases = [{'task': 'reranking_probe', 'variant': variant,
              'result': {'query_count': 8, 'mean_query_end_to_end_seconds': .1,
                         'candidate_count_mean': 20, 'by_language': {
                             'en': {'mean_query_end_to_end_seconds': .1},
                             'zh': {'mean_query_end_to_end_seconds': .2}}}}
             for variant in ('V0', 'V3')]
    result = _evaluation_cost_projection(cases, plan)
    assert result['projected_reranking_seconds'] == 180.0
