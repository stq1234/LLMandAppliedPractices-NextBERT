"""Print the planned training commands; run only with explicit --execute."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def build_commands(model_dir: Path, data_dir: Path, output: Path):
    plan = json.loads((ROOT / 'configs/experiment_plan.json').read_text(encoding='utf-8'))
    train = ROOT / 'src/train_nextbert.py'
    commands = []
    final = {}
    prefix = None
    for variant in ('V0', 'V1', 'V2', 'V3'):
        previous = prefix if variant == 'V3' else None
        for length, absolute_step in ((512, 256), (1024, 512), (4096, 640), (8192, 704)):
            if variant == 'V3' and length < 4096:
                continue
            stage_out = output / 'mlm' / variant / str(length)
            cmd = [sys.executable, str(train), '--task', 'mlm', '--variant', variant,
                   '--model-dir', str(model_dir), '--data', str(data_dir / f'wiki_{length}.npy'),
                   '--length', str(length), '--steps', str(absolute_step),
                   '--schedule-total-steps', '704', '--seed', '17', '--output-dir', str(stage_out)]
            if previous:
                cmd += ['--resume-from', str(previous)]
            if variant == 'V3':
                cmd += ['--mode', 'balanced']
                if length == 4096:
                    cmd += ['--shared-prefix-resume']
            if length == 8192:
                cmd += ['--final-weights-only']
            commands.append(cmd)
            previous = stage_out / f'{variant}_mlm_checkpoint.pt'
            if variant == 'V2' and length == 1024:
                prefix = previous
        final[variant] = previous
    qa_rows = [json.loads(line) for line in (data_dir / 'qa_train.jsonl').read_text(encoding='utf-8').splitlines() if line]
    n_long = sum(len(r['input_ids']) > 512 or r.get('window_type') == 'long' for r in qa_rows)
    qa_steps = math.ceil((len(qa_rows) - n_long) * 2 / 32) + math.ceil(n_long * 2 / 8)
    steps = {'classification': 250, 'qa': qa_steps, 'retrieval': 125, 'reranking': 500}
    for group in plan['downstream']['run_matrix']:
        for variant in group['variants']:
            for task in group['tasks']:
                for seed in group['seeds']:
                    data_task = 'retrieval' if task == 'reranking' else task
                    cmd = [sys.executable, str(train), '--task', task, '--variant', variant,
                           '--model-dir', str(model_dir), '--data', str(data_dir / f'{data_task}_train.jsonl'),
                           '--steps', str(steps[task]), '--seed', str(seed), '--init-from', str(final[variant]),
                           '--final-weights-only',
                           '--output-dir', str(output / 'downstream' / variant / task / str(seed))]
                    if task == 'retrieval':
                        cmd += ['--physical-document-batch', '4']
                    commands.append(cmd)
    return commands


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, default=ROOT / 'artifacts/mmbert-base')
    p.add_argument('--data-dir', type=Path, default=ROOT / 'data/prepared/full')
    p.add_argument('--output-dir', type=Path, default=ROOT / 'results/formal')
    p.add_argument('--gpu', type=int, default=7)
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    commands = build_commands(args.model_dir, args.data_dir, args.output_dir)
    if args.execute:
        needed = {Path(cmd[cmd.index('--data') + 1]) for cmd in commands}
        missing = [str(path) for path in needed if not path.is_file()]
        if missing:
            raise FileNotFoundError(f'Prepared training files missing: {missing}')
        if args.output_dir.exists() and any(args.output_dir.rglob('*checkpoint.pt')):
            raise FileExistsError('Output contains checkpoints; use a fresh directory or resume individual stages explicitly')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    for cmd in commands:
        print(shlex.join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=ROOT, env=env, check=True)
            # A successful next stage supersedes its optimizer checkpoint.
            # V2's step-512 prefix remains available until V3 consumes it.
            if '--resume-from' in cmd:
                previous = Path(cmd[cmd.index('--resume-from') + 1]).resolve()
                variant = cmd[cmd.index('--variant') + 1]
                length = int(cmd[cmd.index('--length') + 1])
                if not (variant == 'V2' and length == 4096):
                    previous.relative_to(args.output_dir.resolve())
                    previous.unlink()
    print(json.dumps({'command_count': len(commands), 'executed': args.execute}))


if __name__ == '__main__':
    main()
