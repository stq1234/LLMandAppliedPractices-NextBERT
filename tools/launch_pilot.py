"""Launch a bounded calibration on one physical GPU, preserving memory headroom."""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=7)
    parser.add_argument('--reserve-gib', type=float, default=6.0)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('pilot_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    os.environ.setdefault('MKL_NUM_THREADS', '4')
    import torch
    free, total = torch.cuda.mem_get_info(0)
    reserve = int(args.reserve_gib * 1024**3)
    if free <= reserve + 4 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory with the requested headroom')
    fraction = min(1.0, (free - reserve) / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    record = {'physical_gpu': args.gpu, 'pid': os.getpid(),
              'free_before_bytes': free, 'total_bytes': total,
              'reserve_bytes': reserve, 'allocator_fraction': fraction,
              'started_unix': time.time(), 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(0),
              'nvidia_smi_before': subprocess.check_output(
                  ['nvidia-smi', '-i', str(args.gpu), '--query-gpu=index,uuid,memory.used,utilization.gpu',
                   '--format=csv,noheader'], text=True).strip()}
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=2), encoding='utf-8')
    forwarded = args.pilot_args[1:] if args.pilot_args[:1] == ['--'] else args.pilot_args
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / 'src'))
    sys.argv = [str(root / 'src' / 'run_pilot.py'), *forwarded]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
