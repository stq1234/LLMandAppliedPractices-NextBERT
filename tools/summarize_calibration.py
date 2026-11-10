"""Combine completed calibration cases without starting GPU work."""
from __future__ import annotations
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_pilot import read_plan, _task_phases, build_time_projection, _evaluation_cost_projection


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def read_lines(name):
    return [json.loads(line) for line in (ROOT / name).read_text(encoding='utf-8').splitlines() if line]


def item(report, name):
    return next(i for i in report['items'] if i['name'] == name)


def main():
    plan = read_plan(ROOT / 'configs/calibration_small_plan.json')
    mlm = read('results/pilot_mlm7/pilot_report.json')['mlm_calibration_cases']
    tasks = read('results/pilot_qa_cls7/pilot_report.json')['downstream_calibration_cases']
    tasks += read_lines('results/pilot_retrieval_final7/pilot_cases.jsonl')
    evaluation = read_lines('results/pilot_index_final7/pilot_cases.jsonl')
    qa_rows = read_lines('data/prepared/full/qa_train.jsonl')
    projection = build_time_projection(mlm, _task_phases(plan), tasks, {'qa_rows': qa_rows}, plan)
    index = _evaluation_cost_projection(evaluation, plan)
    assert not projection['adaptation_projection_is_partial']
    assert not projection['downstream_projection_is_partial']
    assert not index['is_partial']
    measured_inference = {v: read(f'results/calibration/calibration_{v}.json') for v in ('V0', 'V3')}
    by_task = {}
    for phase in projection['downstream_phase_projections']:
        by_task[phase['task']] = by_task.get(phase['task'], 0) + phase['projected_seconds']
    inference = 0.0
    # Actual run matrix: classification 2x V0/V3; QA 3x each architecture family.
    validation = read('results/data_validation_full.json')['counts']
    for variant in ('V0', 'V3'):
        for task, count in (('classification', 2), ('qa', 3)):
            probe = item(measured_inference[variant], f'{task}_official_split_inference')
            dev_ratio = validation[f'{task}_dev']['rows'] / validation[f'{task}_test']['rows']
            inference += probe['estimated_full_split_seconds_linear'] * count * (1 + dev_ratio)
    long_rates = {v: item(read(f'results/calibration/long4_{v}.json'), 'controlled_long_evidence_inference') for v in ('V0', 'V3')}
    long100 = 2 * sum(x['estimated_plan_groups_current_variant_seconds'] for x in long_rates.values())
    components = {
        'mlm_training': projection['adaptation_projected_seconds_from_pilot'],
        **{f'{task}_training': seconds for task, seconds in by_task.items()},
        'index_build_four_variants': index['projected_index_seconds_for_four_variants'],
        'retrieval_dev_and_test_queries': index['projected_retrieval_query_encode_seconds_for_four_variants'],
        'reranking_dev_and_test': index['projected_reranking_seconds'],
        'classification_and_qa_dev_test_inference': inference,
        'retention_mlm_probes_and_latency_reserve': 360.0,
        'checkpoint_io_and_process_startup_reserve': 540.0,
    }
    base = sum(components.values())
    payload = {
        'status': 'calibration_completed_awaiting_training_decision',
        'formal_full_matrix_started': False, 'physical_gpu': 7,
        'components_seconds': components,
        'projection': projection, 'index_projection': index,
        'long_context_projection_seconds_100_groups_per_language': long100,
        'long_context_projection_method': 'batch4, 4 real source groups/language; V0 rate represents V0/V1, V3 rate represents V2/V3',
        'variants_covered_in_optimizer_calibration': ['V0','V1','V2','V3'],
        'sampling_note': 'reranking timing uses 8 total sampled queries, balanced across languages',
        'data_version_note': 'QA optimizer timing preceded exact character-offset repair and train-only filler regeneration; counts/batch/length limits are unchanged. No quality claims are made from calibration heads.',
        'budget_options': {str(groups): {
            'source_groups_per_language': groups,
            'projected_gpu_hours': (base + long100 * groups / 100) / 3600,
            'gpu_hours_with_25_percent_margin': (base + long100 * groups / 100) / 3600 * 1.25,
        } for groups in (25, 100)},
        'unmeasured_reserves': ['held-out MLM inference and latency sweep', 'checkpoint IO/startup beyond the measured single save/resume'],
        'scale_interpretation': 'The implemented 11.53M-token branch is a small architecture-validation run, not evidence of convergence or sufficient formal adaptation.',
        'larger_mlm_only_scenarios': {str(tokens): {
            'input_tokens_per_full_variant_branch': tokens,
            'same_downstream_subsets_and_25_long_groups_per_language': True,
            'projected_gpu_hours_with_25_percent_margin':
                ((base - components['mlm_training'] + components['mlm_training'] * tokens / plan['adaptation']['full_branch_input_tokens']
                  + long100 * .25) / 3600 * 1.25),
            'status': 'unmeasured linear token-throughput extrapolation; larger-batch calibration and additional source data not prepared',
        } for tokens in (50_000_000, 100_000_000)},
    }
    out = ROOT / 'results/calibration_summary.json'
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'components_minutes': {k: round(v / 60, 2) for k,v in components.items()},
                      'long100_minutes': round(long100 / 60, 2), 'budget_options': payload['budget_options']}, indent=2))


if __name__ == '__main__':
    main()
