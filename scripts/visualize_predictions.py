"""
Qualitative check: render predicted vs ground-truth hands on validation frames.

Millimetre metrics say how far off a model is but not how it fails. This draws
both skeletons on the same frame -- ground truth in teal, prediction in amber --
so the failure mode is visible: whether fingers are wrong, whether the hand is in
the right place, or whether the model missed the hand entirely.

Usage:
    python scripts/visualize_predictions.py <ckpt> --out vis.jpg --n 6
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath('.'))

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from hawor.configs import get_config
from lib.core import constants
from lib.datasets.hawor_full_dataset import HaworFullDataset
from lib.models.hawor import load_checkpoint
from lib.models.hawor_full import HaworFull

BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
         (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
         (0, 17), (17, 18), (18, 19), (19, 20)]
GT_COLOR = (140, 125, 15)      # teal, BGR
PRED_COLOR = (18, 90, 168)     # amber, BGR


def denorm(t):
    """Model-space tensor -> BGR uint8 image."""
    mean = np.array(constants.IMG_NORM_MEAN, np.float32).reshape(3, 1, 1)
    std = np.array(constants.IMG_NORM_STD, np.float32).reshape(3, 1, 1)
    img = (t.cpu().numpy() * std + mean) * 255.0
    return np.ascontiguousarray(img.transpose(1, 2, 0)[:, :, ::-1].clip(0, 255).astype(np.uint8))


def draw(img, pts, color, thick=2, dot=3):
    for a, b in BONES:
        cv2.line(img, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), color,
                 thick, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, tuple(p.astype(int)), dot, color, -1, cv2.LINE_AA)


def label(img, text, y, color=(255, 255, 255)):
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--out', default='vis.jpg')
    ap.add_argument('--n', type=int, default=6, help='frames per dataset')
    ap.add_argument('--stride', type=int, default=97, help='spacing between sampled windows')
    args = ap.parse_args()

    d = os.path.dirname(os.path.dirname(os.path.abspath(args.ckpt)))
    cfg = get_config(os.path.join(d, 'model_config.yaml'), merge=True, update_cachedir=False)
    cfg.defrost()
    cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS = ''
    cfg.MODEL.BACKBONE.FP8 = False
    cfg.MODEL.WARM_START = None
    cfg.freeze()

    model = HaworFull(cfg).cuda().eval()
    sd = load_checkpoint(args.ckpt)['state_dict']
    model.load_state_dict(sd, strict=False)

    tiles = []
    for name, root in (('HOT3D', 'datasets/hot3d_clips_export'),
                       ('ARCTIC', 'datasets/arctic_export')):
        ds = HaworFullDataset(root, 'val.json', cfg, seq_len=16, stride=64, train=False)
        picks = list(range(0, len(ds), max(1, args.stride)))[:args.n]
        for wi in picks:
            item = ds[wi]
            batch = {k: v.unsqueeze(0).cuda() for k, v in item.items()}
            out = model.forward_step(batch, train=False)
            t = 8                                    # mid-window frame
            img = denorm(item['img'][t])
            H, W = img.shape[:2]
            scale = np.array([W, H], np.float32)

            gt2 = item['gt_j2d'][t].numpy()           # (2,J,2) normalized
            conf = item['gt_j2d_conf'][t].numpy()
            valid = item['gt_valid'][t].numpy()
            pr2 = out['pred_keypoints_2d'].reshape(16, 2, 21, 2)[t].cpu().numpy()
            vis = torch.sigmoid(out['pred_vis'].reshape(16, 2)[t]).cpu().numpy()

            shown = []
            for slot, hand in ((0, 'L'), (1, 'R')):
                if valid[slot] > 0 and conf[slot].mean() > 0.5:
                    draw(img, (gt2[slot] + 0.5) * scale, GT_COLOR)
                    draw(img, (pr2[slot] + 0.5) * scale, PRED_COLOR)
                    shown.append(f'{hand} vis={vis[slot]:.2f}')
            label(img, f'{name}  w{wi}  ' + ' '.join(shown) if shown else f'{name}  w{wi}', 20)
            tiles.append(img)

    if not tiles:
        raise SystemExit('no frames rendered')
    h = min(t.shape[0] for t in tiles)
    tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
    per_row = 3
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        while len(row) < per_row:
            row.append(np.zeros_like(row[0]))
        rows.append(np.concatenate(row, axis=1))
    w = min(r.shape[1] for r in rows)
    grid = np.concatenate([r[:, :w] for r in rows], axis=0)
    legend = np.zeros((34, grid.shape[1], 3), np.uint8)
    cv2.putText(legend, 'ground truth', (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(legend, 'prediction', (170, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, PRED_COLOR, 2, cv2.LINE_AA)
    grid = np.concatenate([legend, grid], axis=0)
    cv2.imwrite(args.out, grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f'wrote {args.out}  {grid.shape[1]}x{grid.shape[0]}  ({len(tiles)} frames)')


if __name__ == '__main__':
    main()
