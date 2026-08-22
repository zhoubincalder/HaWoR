"""
MPJPE / PA-MPJPE in mm for the full-frame two-hand model (lib/models/hawor_full.py).

Kept separate from scripts/eval_mpjpe.py because the interfaces differ: the crop
model returns one hand per sample, this one returns a (sample, 2) slot layout with
a per-hand validity mask, and absent hands must be excluded rather than scored.

Metrics are computed over valid hands only, and also reported per hand so a model
that has learned one hand better than the other is visible rather than averaged
away. Visibility is scored separately as a classification accuracy, since on full
frames predicting hand presence is part of the task.

Usage:
    python scripts/eval_mpjpe_full.py logs/full_sapiens2_08b/checkpoints/best-*.ckpt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import torch
from torch.utils.data import DataLoader

from hawor.configs import get_config
from lib.datasets.hawor_full_dataset import HaworFullDataset
from lib.models.hawor import load_checkpoint
from lib.models.hawor_full import HaworFull

SETS = {'HOT3D': 'datasets/hot3d_clips_export', 'ARCTIC': 'datasets/arctic_export'}


def procrustes(S1, S2):
    """Similarity-align S1 onto S2, both (B,J,3)."""
    mu1, mu2 = S1.mean(1, keepdim=True), S2.mean(1, keepdim=True)
    X1, X2 = S1 - mu1, S2 - mu2
    var1 = (X1 ** 2).sum(dim=(1, 2), keepdim=True)
    U, s, Vh = torch.linalg.svd(X1.transpose(1, 2) @ X2)
    V = Vh.transpose(1, 2)
    sign = torch.sign(torch.linalg.det(U @ V.transpose(1, 2)))
    Z = torch.eye(3, device=S1.device).unsqueeze(0).repeat(S1.shape[0], 1, 1)
    Z[:, -1, -1] = sign
    R = V @ Z @ U.transpose(1, 2)
    # Only the last singular value flips under a reflection.
    s_signed = s.clone()
    s_signed[:, -1] = s_signed[:, -1] * sign
    scale = s_signed.sum(1).view(-1, 1, 1) / var1
    return scale * (R @ X1.transpose(1, 2)).transpose(1, 2) + mu2


@torch.no_grad()
def evaluate(ckpt, batch_size, workers):
    d = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    cfg = get_config(os.path.join(d, 'model_config.yaml'), merge=True, update_cachedir=False)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ''
    # Evaluate in plain precision. If the model is built with fp8, its weights are
    # torchao quantized subclasses and copying a plain checkpoint tensor into one
    # fails with "'Tensor' object has no attribute 'qdata'". Weights come from the
    # checkpoint anyway, so quantization only affects speed here.
    cfg.MODEL.BACKBONE.FP8 = False
    cfg.freeze()

    model = HaworFull(cfg).cuda().eval()
    sd = load_checkpoint(ckpt)['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  WARNING {len(missing)} missing keys e.g. {list(missing)[:3]}')
    if unexpected:
        print(f'  WARNING {len(unexpected)} unexpected keys e.g. {list(unexpected)[:3]}')

    res = {}
    for name, root in SETS.items():
        ds = HaworFullDataset(root, 'val.json', cfg, seq_len=16,
                              stride=cfg.TRAIN.get('VAL_CHUNK_STRIDE', 64), train=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        acc = {k: [0.0, 0.0, 0] for k in ('all', 'left', 'right')}   # mp, pa, n
        vis_ok = vis_n = 0
        for batch in dl:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            o = model.forward_step(batch, train=False)
            pred = o['pred_keypoints_3d'].float()               # (B*T,2,J,3)
            gt = batch['gt_j3d_wo_trans'].flatten(0, 1).float()  # (B*T,2,J,3)
            valid = batch['gt_valid'].flatten(0, 1) > 0          # (B*T,2)

            # visibility accuracy
            vis_ok += int(((o['pred_vis'] > 0) == valid).sum())
            vis_n += valid.numel()

            for slot, key in ((0, 'left'), (1, 'right')):
                m = valid[:, slot]
                if not bool(m.any()):
                    continue
                p = pred[m, slot]
                g = gt[m, slot]
                p = p - p[:, :1]
                g = g - g[:, :1]
                mp = float((p - g).norm(dim=-1).mean()) * 1000
                pa = float((procrustes(p, g) - g).norm(dim=-1).mean()) * 1000
                k = int(m.sum())
                for tgt in (key, 'all'):
                    acc[tgt][0] += mp * k
                    acc[tgt][1] += pa * k
                    acc[tgt][2] += k
        res[name] = {k: (v[0] / v[2], v[1] / v[2]) if v[2] else (float('nan'),) * 2
                     for k, v in acc.items()}
        res[name]['vis_acc'] = 100.0 * vis_ok / max(vis_n, 1)
    del model
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    for c in args.ckpts:
        print(f'\n=== {os.path.basename(c)} ===', flush=True)
        r = evaluate(c, args.batch_size, args.workers)
        print(f'{"dataset":<8} {"MPJPE":>7} {"PA":>7} {"L MPJPE":>8} {"L PA":>7} '
              f'{"R MPJPE":>8} {"R PA":>7} {"vis%":>6}')
        for name, v in r.items():
            print(f'{name:<8} {v["all"][0]:>7.2f} {v["all"][1]:>7.2f} '
                  f'{v["left"][0]:>8.2f} {v["left"][1]:>7.2f} '
                  f'{v["right"][0]:>8.2f} {v["right"][1]:>7.2f} {v["vis_acc"]:>6.1f}', flush=True)


if __name__ == '__main__':
    main()
