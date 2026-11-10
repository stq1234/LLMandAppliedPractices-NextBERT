"""Evaluate held-out MLM with fixed masks and target-count weighted loss."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
import torch
from evaluate_nextbert import _load_checkpoint
from train_nextbert import load_nextbert, corrupt_mlm, _amp_context, _special_ids


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--variant', choices=['V0', 'V1', 'V2', 'V3'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seed', type=int, default=1701)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    if a.batch_size < 2:
        raise ValueError('use a physical batch of at least two')
    rows = np.load(a.data, mmap_mode='r')
    meta = [json.loads(line) for line in a.data.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if len(meta) != len(rows) or any(r.get('split') not in {'dev', 'test'} for r in meta):
        raise ValueError('MLM retention evaluation requires held-out split provenance')
    device = torch.device(a.device)
    model = load_nextbert(a.model_dir, a.variant, device)
    if a.checkpoint:
        _load_checkpoint(model, a.checkpoint, device)
    model.eval()
    nll = {}; counts = {}; started = time.perf_counter()
    # Corrupt each row with its stable index, independently of batch size.
    for offset in range(0, len(rows), a.batch_size):
        selected = []
        for index in range(offset, min(offset + a.batch_size, len(rows))):
            ids = torch.from_numpy(np.array(rows[index:index + 1], dtype=np.int64)).to(device)
            masked, labels, targets = corrupt_mlm(ids, ids.ne(0), vocab_size=model.config.vocab_size,
                special_ids=_special_ids(), mask_token_id=4, seed=a.seed + index, mode='token')
            selected.append((index, masked, labels, int(targets.sum())))
        # Separate language minibatches preserves exact token-weighted reporting.
        for language in sorted({str(meta[index]['language']) for index, *_ in selected}):
            chunk = [r for r in selected if str(meta[r[0]]['language']) == language]
            ids = torch.cat([r[1] for r in chunk]); labels = torch.cat([r[2] for r in chunk])
            target_count = sum(r[3] for r in chunk)
            with torch.inference_mode(), _amp_context(device):
                loss = model.forward_mlm(ids, ids.ne(0), labels=labels)['loss']
            nll[language] = nll.get(language, 0.0) + float(loss.float()) * target_count
            counts[language] = counts.get(language, 0) + target_count
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    result = {'task': 'fixed_mask_heldout_mlm', 'variant': a.variant, 'seed': a.seed,
              'rows': len(rows), 'length': rows.shape[1], 'mask_rate': .30,
              'mean_nll_per_target': sum(nll.values()) / sum(counts.values()),
              'by_language': {k: {'targets': counts[k], 'mean_nll_per_target': nll[k] / counts[k]} for k in nll},
              'elapsed_seconds': time.perf_counter() - started, 'data': str(a.data),
              'checkpoint': str(a.checkpoint) if a.checkpoint else None}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
