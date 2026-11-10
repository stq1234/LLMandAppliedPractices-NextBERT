"""Validate split isolation and interfaces of actual prepared data."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def rows(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def validate(root, runtime_only=False):
    summary = {}
    for length in (512, 1024, 4096, 8192):
        article_splits = {}
        for split in ('train', 'dev', 'test'):
            name = f'wiki_{length}' + (f'_{split}' if split != 'train' else '')
            array = np.load(root / f'{name}.npy', mmap_mode='r')
            meta = rows(root / f'{name}.jsonl')
            assert array.dtype == np.int32 and array.shape == (len(meta), length) and len(meta) > 0, name
            assert np.all(array[:, 0] == 2) and np.all(array[:, -1] == 1), name
            for row in meta:
                assert row['split'] == split, name
                source = row['source_id']
                assert source not in article_splits or article_splits[source] == split, source
                article_splits[source] = split
            summary[name] = dict(Counter(row['language'] for row in meta))
    for task in ('classification', 'qa', 'retrieval'):
        split_ids = {}
        for split in ('train', 'dev', 'test'):
            name = f'{task}_{split}'
            path = root / f'{name}.jsonl'
            if runtime_only and task == 'retrieval' and split != 'train':
                path = root / f'retrieval_queries_{split}.jsonl'
            data = rows(path)
            assert data, name
            for row in data:
                key = str(row.get('question_id', row['id']))
                assert key not in split_ids or split_ids[key] == split, (name, key)
                split_ids[key] = split
                if task == 'qa':
                    n = len(row['input_ids'])
                    assert len(row['context_mask']) == len(row['offset_mapping']) == n
                    start, end = row['start_positions'], row['end_positions']
                    if start != -100:
                        assert 0 <= start <= end < n and row['context_mask'][start] and row['context_mask'][end]
                        a, b = row['offset_mapping'][start][0], row['offset_mapping'][end][1]
                        gold_a, gold_b = row['answer_start_char'], row['answer_end_char']
                        assert a <= gold_a < gold_b <= b, (name, key)
                        assert row['context'][gold_a:gold_b] == row['answer_text'], (name, key)
                if task == 'retrieval' and split == 'train':
                    assert len(row['query_ids']) <= 128 and row['positive_ids'] and row['negative_ids']
                    negatives = row['negative_candidates']
                    assert len(negatives) >= 2
                    assert len({tuple(row['positive_ids']), *(tuple(r['input_ids']) for r in negatives[:2])}) == 3
            summary[name] = {'rows': len(data), 'languages': dict(Counter(row['language'] for row in data))}
    corpus = rows(root / 'retrieval_corpus.jsonl')
    corpus_ids = {row['id'] for row in corpus}
    assert len(corpus_ids) == len(corpus)
    project = Path(__file__).resolve().parents[1]
    profile = root.name if root.name != 'prepared' else 'pilot'
    backgrounds = project / 'artifacts/raw' / profile / 'wikipedia'
    if backgrounds.is_dir():
        train_sources = {}
        for lang in ('en', 'zh'):
            train_sources[lang] = {str(r.get('source_id') or r.get('id'))
                for r in rows(backgrounds / f'selected_{lang}.jsonl') if r.get('split') == 'train'}
        for row in rows(root / 'qa_train.jsonl'):
            if row.get('window_type') == 'long':
                assert set(row['background_source_ids']) <= train_sources[row['language']]
    for split in ('dev', 'test'):
        for query in rows(root / f'retrieval_queries_{split}.jsonl'):
            assert query['input_ids'] and set(query['relevant_doc_ids']) <= corpus_ids
        for group in rows(root / f'reranking_{split}.jsonl'):
            assert len(group['candidate_ids']) == len(group['candidate_docids']) == 20
    for manifest in root.glob('manifest_*.json'):
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        project = Path(__file__).resolve().parents[1]
        plan_reference = payload.get('experiment_plan')
        if isinstance(plan_reference, dict) and plan_reference.get('snapshot_path'):
            snapshot = project / plan_reference['snapshot_path']
            with snapshot.open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == plan_reference['sha256'], str(snapshot)
        for item in payload.get('artifacts', {}).values():
            if isinstance(item, dict) and item.get('path') and item.get('sha256'):
                path = project / item.get('snapshot_path', item['path'])
                if path.exists():
                    with path.open('rb') as stream:
                        assert hashlib.file_digest(stream, 'sha256').hexdigest() == item['sha256'], str(path)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data_dir', type=Path)
    p.add_argument('--runtime-only', action='store_true', help='Use compact retrieval dev/test query interfaces instead of source-passage copies')
    args = p.parse_args()
    print(json.dumps({'validated': True, 'counts': validate(args.data_dir, args.runtime_only)}, indent=2))
