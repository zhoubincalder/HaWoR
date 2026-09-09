"""
Smoke-test every converted dataset through the real training dataset class.

The converters verify labels against each source's own ground truth, to ~1e-4mm.
That is not the same as verifying the training pipeline can consume them: a fold
can load and still yield unusable batches -- images that do not exist, joints
projected outside the frame, a principal point that survives conversion but not
the letterbox rescale, all-invalid windows. This checks what the model would
actually receive.

Per dataset, per fold: build HaworFullDataset exactly as train_full.py does,
then draw sample windows and assert on the tensors the model consumes.

What each check is for:

  windows          a fold with zero windows raises SystemExit in training; here
                   it is reported per fold instead of killing the run
  finite           NaN or Inf in any tensor. mano_trans arrived NaN from bronze,
                   so this is not hypothetical
  valid frames     a window where no hand is valid trains on nothing
  j2d in frame     gt_j2d is normalized to [-0.5, 0.5] over the input frame.
                   Confident joints landing outside mean the projection and the
                   letterbox disagree -- the failure mode a non-central
                   principal point would cause
  focal / centre   focal must be positive and the centre inside the frame after
                   `f*s`, `c*s + pad`
  j3d scale        wrist-relative joint norms should sit in hand range, a few cm.
                   Catches a metre/millimetre error that leaves everything
                   finite and plausible-looking
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath('.'))

import numpy as np
import torch

DATASETS = [
    ('hot3d', 'datasets/hot3d_clips_export'),
    ('dexycb', 'datasets/dexycb_bronze_export'),
    ('arctic', 'datasets/arctic_export'),
    ('ho3d', 'datasets/ho3d_export'),
    ('h2o', 'datasets/h2o_export'),
    ('h2o3d', 'datasets/h2o3d_export'),
]


def check_window(b, in_w, in_h):
    """Return a list of problem strings for one sample."""
    bad = []
    for k, v in b.items():
        if not torch.isfinite(v).all():
            n = int((~torch.isfinite(v)).sum())
            bad.append(f'{k}: {n} non-finite')

    valid = b['gt_valid']                        # (T,2)
    if valid.sum() == 0:
        bad.append('no valid hand in any frame')

    # Confident joints must land inside the normalized frame.
    conf = b['gt_j2d_conf'] > 0                  # (T,2,J)
    if conf.any():
        j2d = b['gt_j2d'][conf]
        out = ((j2d < -0.5) | (j2d > 0.5)).any(-1).sum()
        frac = float(out) / max(int(conf.sum()), 1)
        if frac > 0.02:
            bad.append(f'{100 * frac:.1f}% of confident joints outside frame')

    f = float(b['img_focal'][0])
    cx, cy = (float(x) for x in b['img_center'][0])
    if not (f > 0):
        bad.append(f'focal {f:.1f} not positive')
    if not (0 <= cx <= in_w and 0 <= cy <= in_h):
        bad.append(f'centre ({cx:.1f},{cy:.1f}) outside {in_w}x{in_h}')

    # Hand extent from the wrist, in metres.
    j3d = b['gt_j3d_wo_trans']                   # (T,2,J,3)
    m = valid.bool()
    if m.any():
        sel = j3d[m]                             # (N,J,3)
        ext = (sel - sel[:, :1]).norm(dim=-1).max().item()
        if not (0.02 < ext < 0.40):
            bad.append(f'hand extent {ext * 1000:.0f}mm outside 20-400mm')

    img = b['img']
    if img.abs().max() > 20:
        bad.append(f'img range suspicious (max abs {img.abs().max():.1f})')
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='hawor/configs/hawor_full_sapiens2_1024.yaml')
    ap.add_argument('--samples', type=int, default=8,
                    help='windows drawn per fold')
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()

    from hawor.configs import get_config
    from lib.datasets.hawor_full_dataset import HaworFullDataset
    cfg = get_config(args.cfg, merge=True)
    in_h = cfg.MODEL.get('INPUT_H', 384)
    in_w = cfg.MODEL.get('INPUT_W', 512)
    print(f'config {args.cfg}: input {in_h}x{in_w}\n')

    rows, failures = [], 0
    for name, root in DATASETS:
        if args.only and name not in args.only:
            continue
        if not os.path.isdir(root):
            print(f'{name}: MISSING {root}')
            failures += 1
            continue
        for fold, stride in (('train', cfg.TRAIN.get('CHUNK_STRIDE', 8)),
                             ('val', cfg.TRAIN.get('VAL_CHUNK_STRIDE', 64)),
                             ('test', cfg.TRAIN.get('VAL_CHUNK_STRIDE', 64))):
            sf = os.path.join(root, f'{fold}.json')
            if not os.path.exists(sf):
                continue
            try:
                ds = HaworFullDataset(root, f'{fold}.json', cfg, seq_len=16,
                                      stride=stride, train=(fold == 'train'))
            except Exception as e:
                print(f'{name}/{fold}: BUILD FAILED {type(e).__name__}: {e}')
                failures += 1
                continue
            if len(ds) == 0:
                print(f'{name}/{fold}: 0 windows')
                failures += 1
                continue
            rng = np.random.default_rng(0)
            idx = rng.choice(len(ds), size=min(args.samples, len(ds)), replace=False)
            probs, nvalid = [], 0
            for i in idx:
                try:
                    b = ds[int(i)]
                except Exception as e:
                    probs.append(f'window {i}: {type(e).__name__}: {e}')
                    continue
                nvalid += int(b['gt_valid'].sum())
                probs += [f'window {i}: {p}' for p in check_window(b, in_w, in_h)]
            status = 'OK' if not probs else f'{len(probs)} PROBLEM(S)'
            rows.append((name, fold, len(ds), nvalid, status))
            print(f'{name:8} {fold:5} {len(ds):7} windows  '
                  f'{nvalid:5} valid hand-frames in {len(idx)} sampled  {status}')
            for p in probs[:6]:
                print(f'    - {p}')
            if probs:
                failures += 1

    print(f'\n{"dataset":9} {"fold":6} {"windows":>9} {"status":>12}')
    for name, fold, n, _, st in rows:
        print(f'{name:9} {fold:6} {n:9} {st:>12}')
    print(f'\ntotal train windows: '
          f'{sum(n for _, f, n, _, _ in rows if f == "train")}')
    if failures:
        print(f'\n{failures} fold(s) with problems')
        return 1
    print('\nall folds OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
