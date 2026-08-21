"""
MPJPE / PA-MPJPE in mm on the held-out val splits, per dataset.

Each checkpoint is evaluated with the config that was saved alongside it
(<logdir>/model_config.yaml), so ViT-H and Sapiens2 runs can be compared without
hand-editing anything. The released weights/hawor/ tree has the same layout.

Usage:
    python scripts/eval_mpjpe.py weights/hawor/checkpoints/hawor.ckpt \
        logs/hawor_fixed/checkpoints/best-*.ckpt
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import torch
from torch.utils.data import ConcatDataset, DataLoader

from hawor.configs import get_config
from lib.datasets.hawor_train_dataset import HaworChunkDataset
from lib.models.hawor import HAWOR, load_checkpoint

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
    # Only the last singular value flips under a reflection; scaling the whole
    # sum by det(Z) would negate the transform and inflate the error.
    s_signed = s.clone()
    s_signed[:, -1] = s_signed[:, -1] * sign
    scale = s_signed.sum(1).view(-1, 1, 1) / var1
    return scale * (R @ X1.transpose(1, 2)).transpose(1, 2) + mu2


def config_for(ckpt):
    d = os.path.dirname(os.path.dirname(os.path.abspath(ckpt)))
    p = os.path.join(d, 'model_config.yaml')
    if not os.path.exists(p):
        raise SystemExit(f'no model_config.yaml next to {ckpt} (looked in {d})')
    return p


@torch.no_grad()
def evaluate(ckpt, batch_size, workers):
    cfg = get_config(config_for(ckpt), merge=True, update_cachedir=False)
    cfg.defrost()
    # Weights come from the checkpoint; skip the redundant backbone preload.
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ''
    cfg.MODEL.LOAD_WEIGHTS = None
    cfg.freeze()

    model = HAWOR(cfg).cuda().eval()
    sd = load_checkpoint(ckpt)['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f'  WARNING {len(missing)} missing keys, e.g. {list(missing)[:3]}')
    if unexpected:
        print(f'  WARNING {len(unexpected)} unexpected keys, e.g. {list(unexpected)[:3]}')

    out = {}
    for name, root in SETS.items():
        ds = HaworChunkDataset(root, 'val.json', cfg, seq_len=16,
                               stride=cfg.TRAIN.get('VAL_CHUNK_STRIDE', 64), train=False)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
        mp = pa = n = 0
        for batch in dl:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
            o = model.forward_step(batch, train=False)
            pred = o['pred_keypoints_3d'].float()
            gt = batch['gt_j3d_wo_trans'].flatten(0, 1).float()
            pred = pred - pred[:, :1]
            gt = gt - gt[:, :1]
            b = pred.shape[0]
            mp += float((pred - gt).norm(dim=-1).mean()) * b * 1000
            pa += float((procrustes(pred, gt) - gt).norm(dim=-1).mean()) * b * 1000
            n += b
        out[name] = (mp / n, pa / n)
    del model
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpts', nargs='+')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    print(f'{"model":<44} {"HOT3D MPJPE":>12} {"HOT3D PA":>9} {"ARCTIC MPJPE":>13} {"ARCTIC PA":>10}',
          flush=True)
    for c in args.ckpts:
        tag = os.path.basename(c).replace('.ckpt', '')[:42]
        r = evaluate(c, args.batch_size, args.workers)
        print(f'{tag:<44} {r["HOT3D"][0]:>11.2f} {r["HOT3D"][1]:>9.2f} '
              f'{r["ARCTIC"][0]:>12.2f} {r["ARCTIC"][1]:>10.2f}', flush=True)


if __name__ == '__main__':
    main()
