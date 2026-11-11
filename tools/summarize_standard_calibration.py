"""Summarize the standard recipe from matching real-batch calibration cases."""
from __future__ import annotations
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from run_pilot import read_plan, _task_phases, build_time_projection, _evaluation_cost_projection


def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


def lines(path):
    return [json.loads(line) for line in (ROOT / path).read_text(encoding='utf-8').splitlines() if line]


def get_item(report, name):
    return next(item for item in report['items'] if item['name'] == name)


def main():
    plan = read_plan()
    mlm_report = read('results/pilot_standard67m_mlm7/pilot_report.json')
    task_report = read('results/pilot_standard67m_tasks7/pilot_report.json')
    mlm = mlm_report['mlm_calibration_cases']
    optimized_span_report = ROOT / 'results/pilot_standard67m_span_optimized7/pilot_report.json'
    optimization_wall_seconds = 0.0
    if optimized_span_report.is_file():
        optimized = json.loads(optimized_span_report.read_text(encoding='utf-8'))
        merged = {(case['variant'], case['length'], case['mode']): case for case in mlm}
        for case in optimized['mlm_calibration_cases']:
            merged[(case['variant'], case['length'], case['mode'])] = case
        mlm = list(merged.values())
        optimization_wall_seconds = optimized['pilot_wall_seconds']
    tasks = task_report['downstream_calibration_cases']
    stage_lookup = {stage['max_length']: stage for stage in plan['adaptation']['stages']}
    for case in mlm:
        stage = stage_lookup[case['length']]
        assert case['physical_microbatch'] == stage['physical_microbatch']
        assert case['gradient_accumulation'] == stage['gradient_accumulation']
        expected_tokens = case['length'] * stage['physical_microbatch'] * stage['gradient_accumulation']
        assert all(row['nonpad_tokens'] == expected_tokens for row in case['steps'])
    projection = build_time_projection(mlm, _task_phases(plan), tasks, {}, plan)
    assert not projection['adaptation_projection_is_partial'], 'Missing matching MLM measurement'
    assert not projection['downstream_projection_is_partial'], 'Missing matching downstream measurement'
    index_cases = lines('results/pilot_index_final7/pilot_cases.jsonl')
    index = _evaluation_cost_projection(index_cases, plan)
    assert not index['is_partial']
    downstream = Counter()
    for phase in projection['downstream_phase_projections']:
        downstream[phase['task']] += phase['projected_seconds']
    # Held-out classification/QA, corpus, and inference batches are unchanged.
    old_validation = read('results/data_validation_full.json')['counts']
    ordinary_inference = 0.0
    inference_sources = []
    long_sample_rates = {}
    inference_runs = Counter()
    for phase in _task_phases(plan):
        family = 'V0' if phase['variant'] in ('V0', 'V1') else 'V3'
        inference_runs[(family, phase['task'])] += 1
    for variant in ('V0', 'V3'):
        inference = read(f'results/calibration/calibration_{variant}.json')
        inference_sources.append(f'results/calibration/calibration_{variant}.json')
        for task in ('classification', 'qa'):
            runs = inference_runs[(variant, task)]
            case = get_item(inference, f'{task}_official_split_inference')
            ratio = old_validation[f'{task}_dev']['rows'] / old_validation[f'{task}_test']['rows']
            ordinary_inference += case['estimated_full_split_seconds_linear'] * runs * (1 + ratio)
        long_case = get_item(read(f'results/calibration/long4_{variant}.json'), 'controlled_long_evidence_inference')
        assert long_case['evaluation_batch_size'] == 4
        long_sample_rates[variant] = long_case['elapsed_seconds'] / long_case['evaluated_source_group_count']
    groups_per_language = plan['evaluation']['long_context']['source_groups_per_language']
    # V0 conservatively represents V1; V2 and V3 use the same architecture.
    long_seconds = sum(long_sample_rates.values()) * 2 * (groups_per_language * 2)
    components = {
        'mlm_training': projection['adaptation_projected_seconds_from_pilot'],
        **{f'{task}_training': seconds for task, seconds in downstream.items()},
        'index_build_four_variants': index['projected_index_seconds_for_four_variants'],
        'retrieval_dev_and_test_queries': index['projected_retrieval_query_encode_seconds_for_four_variants'],
        'reranking_dev_and_test': index['projected_reranking_seconds'],
        'classification_and_qa_dev_test_inference': ordinary_inference,
        'long_context_50_groups_per_language': long_seconds,
        'retention_mlm_probes_and_latency_reserve': 600.0,
        'checkpoint_io_and_startup_reserve': 720.0,
    }
    total = sum(components.values())
    payload = {
        'status': 'standard_recipe_real_batch_calibration_completed',
        'formal_full_matrix_started': False,
        'gpu': 'physical GPU7, RTX A6000 48GB, one process',
        'input_tokens_per_full_branch': plan['adaptation']['full_branch_input_tokens'],
        'executed_mlm_input_tokens_including_shared_prefix': plan['adaptation']['total_input_tokens'],
        'adaptation_steps': plan['adaptation']['total_executed_steps'],
        'downstream_training_phases': plan['downstream']['total_training_phases'],
        'components_seconds': components,
        'projected_gpu_hours': total / 3600,
        'gpu_hours_with_25_percent_margin': total / 3600 * 1.25,
        'optimizer_projection': projection,
        'index_projection': index,
        'long_context_source_seconds_per_group': long_sample_rates,
        'long_context_projection_note': 'batch4; 4 source groups/language measured; V0 rate represents V0/V1, V3 rate represents V2/V3',
        'ordinary_inference_sources': inference_sources,
        'measurement_vs_projection': 'Real short optimizer runs use the final candidate batches. Full durations multiply median optimizer-step times by planned steps; long evaluation and index costs use real sample rates. IO/startup and retention/latency are explicit reserves.',
        'calibration_data_scope': 'The preceding frozen full profile supplies real representative examples; the larger full_67m training data is prepared independently. Timing is not evidence of convergence or model superiority.',
        'maximum_allocated_gib': max(row['memory']['peak_allocated_bytes'] for case in mlm + tasks for row in case['steps']) / 2**30,
        'completed_calibration_wall_seconds': mlm_report['pilot_wall_seconds'] + task_report['pilot_wall_seconds'] + optimization_wall_seconds,
        'span_masking_implementation': 'linear candidate scan with pre-sampled geometric spans; optimized span cases supersede prior quadratic-scanning cases when present',
    }
    output = ROOT / 'results/calibration_standard67m_summary.json'
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')
    print(json.dumps({
        'components_minutes': {k: round(value / 60, 2) for k,value in components.items()},
        'projected_gpu_hours': round(total / 3600, 3),
        'gpu_hours_with_25_percent_margin': round(total / 3600 * 1.25, 3),
        'maximum_allocated_gib': round(payload['maximum_allocated_gib'], 3)}, indent=2))


if __name__ == '__main__':
    main()
