"""
Per dataset: what does training at native resolution cost in accuracy?

MODEL.NATIVE_RES is a 2.15x speed win, but every number behind it so far is
throughput. The accuracy question is not the same for every dataset, which is
why this is per-dataset rather than one figure:

  * DexYCB, HO3D, H2O3D are natively 640x480, so the canvas was interpolating
    them UP 1.6x. Native discards no information at all -- but it does shrink
    the hand's TOKEN budget (DexYCB: 9.3 -> 5.8 tokens), and sub-patch precision
    then has to come from inside a token. This is the case that decides it.
  * HOT3D (704x704) and ARCTIC (840x600) barely move, 1.09x and 1.22x.
  * H2O is the only source LARGER than the canvas, so native scales it DOWN and
    genuinely throws pixels away.

Three numbers per dataset, all in millimetres or source pixels so they compare
across resolutions:

  shift        how far native moves predictions from the canvas ones. Same
               methodology as probe_token_reduction.py, so it is directly
               comparable to mid-trunk pooling's 15.73mm and 640x480's 5.70mm.
               An UPPER bound on the trained-from-here cost: the checkpoint was
               trained at the canvas, and a fine-tune re-adapts the head.
  MPJPE        wrist-relative error against ground truth, for BOTH inputs. This
               is the accuracy that matters, but it is biased toward the canvas,
               which is the resolution this checkpoint trained at. Read the
               DIFFERENCE between the two columns, not either one alone.
  2D           reprojection error against ground truth in SOURCE pixels. The
               normalized coordinates are relative to each run's own frame, so
               they are scaled back through that run's letterbox (dn * out / s)
               to land in a unit both runs share.

MEASURED on the frozen-trunk checkpoint (see the caveat below):

  dataset   source    canvas   native   shift   MPJPE canvas  MPJPE native
  hot3d    704x704   3072tok  1936tok   9.55mm      26.45mm       26.92mm
  dexycb   640x480   3072tok  1200tok  15.04mm     138.58mm      140.26mm
  arctic   840x600   3072tok  2014tok   4.67mm      18.69mm       18.70mm
  ho3d     640x480   3072tok  1200tok  10.61mm     137.01mm      135.34mm
  h2o     1280x720   3072tok  2304tok  11.18mm      20.97mm       24.47mm
  h2o3d    640x480   3072tok  1200tok  11.13mm     120.05mm      117.85mm
  weighted                             11.13mm      83.92mm       84.63mm

Weighted MPJPE delta +0.71mm, with per-dataset deltas mixed in sign (native is
BETTER on ho3d and h2o3d). So resolution is not what limits this model.

Two things the table does not say on its own:

1. Most of "shift" is the PADDING, not the resolution. H2O isolates it: its
   canvas content is 1024x576 letterboxed into 1024x768, and its native input is
   1024x576 with no padding -- pixel-identical content, differing only by the
   black bars. That alone moves predictions 11.18mm and costs 3.50mm of MPJPE.
   The trained model uses the letterbox padding, the same way the masking probe
   found it uses the background.

2. The 120-140mm MPJPE on dexycb / ho3d / h2o3d is NOT a label bug and NOT a
   resolution effect -- it is there in both columns. Ruled out: handedness
   (GT chirality is identical across all six datasets) and scale (wrist to
   index-MCP is 85-95mm everywhere). The cause is that those three are far
   harder and far from what this checkpoint saw. Model-free, on wrist-relative
   GT, mean distance to the dataset's own mean pose and to HOT3D's mean pose:

     hot3d 52.9 / 52.9    dexycb 86.1 / 120.9    arctic 41.7 /  58.9
     ho3d  94.1 / 144.8   h2o    30.9 /  67.9    h2o3d  99.0 / 127.8

   On those three the model scores at or slightly worse than the constant-pose
   baseline, i.e. it contributes nothing -- a pure generalization gap for a head
   trained on HOT3D. Their MPJPE columns are therefore not accuracy measures;
   only the canvas-vs-native DELTA is, since that holds data and weights fixed.

CAVEAT on every number here, and on the other probes in this directory: this
checkpoint's trunk is bit-identical to pretrained Sapiens2 (613 tensors, max abs
diff 0.0), so the backbone was FROZEN and only proj + head + motion trained. It
is a weak instrument for predicting what a full fine-tune does. What partly
rescues it is that the frozen trunk turns out to be remarkably resolution-flat
on HOT3D val -- MPJPE 26.98 / 26.05 / 26.45 / 27.38mm and loss 0.0796 / 0.0793 /
0.0804 / 0.0820 at 768 / 1728 / 3072 / 4800 tokens -- so 768x1024 is not a
special operating point and the probe references are not off-distribution.
Native's loss is the outlier at 0.0904 (+12%) while its MPJPE barely moves,
which again points at the 2D/translation terms, i.e. the padding.
"""
import argparse
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


def source_size(root):
    """(W0, H0) of a dataset, from the first sequence that has an annotation."""
    for d in sorted(os.listdir(root)):
        f = os.path.join(root, d, 'train_anno.npz')
        if os.path.exists(f):
            a = np.load(f, allow_pickle=True)
            return int(a['img_size'][0]), int(a['img_size'][1])
    raise RuntimeError(f'no train_anno.npz under {root}')


def build_cfg(cfg_path, native):
    from hawor.configs import get_config
    cfg = get_config(cfg_path, merge=True, update_cachedir=True)
    cfg.defrost()
    cfg.MODEL.INPUT_H, cfg.MODEL.INPUT_W = 768, 1024
    cfg.MODEL.NATIVE_RES = native
    cfg.MODEL.RES_JITTER_LEVELS = []       # eval: no jitter, so runs line up
    cfg.TRAIN.BATCH_SIZE = 1               # NATIVE_RES asserts this
    cfg.MODEL.BACKBONE.FREEZE = True
    cfg.MODEL.BACKBONE.GRAD_CHECKPOINT = False
    cfg.MODEL.BACKBONE.TORCH_COMPILE = 0
    cfg.MODEL.BACKBONE.FP8 = cfg.MODEL.BACKBONE.FP8_TRAINING = False
    cfg.MODEL.WARM_START = ''
    cfg.freeze()
    return cfg


def wrist_relative(p, gt):
    """Wrist-relative per-joint error in mm, matching the 3D loss's pelvis_id=0."""
    return ((p - p[:, :1]) - (gt - gt[:, :1])).norm(dim=-1) * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--cfg', default='hawor/configs/hawor_full_sapiens2_1024.yaml')
    ap.add_argument('--n', type=int, default=16, help='val windows per dataset')
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()

    from lib.datasets.hawor_full_dataset import HaworFullDataset
    from lib.models.hawor_full import HaworFull

    cfg_c, cfg_n = build_cfg(args.cfg, False), build_cfg(args.cfg, True)
    # ONE model: native resolution changes only the input tensor. forward_step
    # takes the frame size from batch['img_size'], so the same weights serve
    # both -- and any difference below is the input, not two loads.
    model = HaworFull(cfg_c)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(sd.get('state_dict', sd), strict=False)
    model = model.cuda().eval()

    print(f'\n{"dataset":8} {"source":>10} {"canvas":>10} {"native":>10} '
          f'{"shift":>9} {"MPJPE canvas":>13} {"MPJPE native":>13} '
          f'{"2D canvas":>10} {"2D native":>10}')
    print('-' * 108)
    rows = []
    for name, root in DATASETS:
        if (args.only and name not in args.only) or not os.path.isdir(root):
            continue
        W0, H0 = source_size(root)
        fit = min(768 / H0, 1024 / W0)
        s_c, s_n = fit, min(1.0, fit)

        ds_c = HaworFullDataset(root, 'val.json', cfg_c, seq_len=16, stride=64, train=False)
        ds_n = HaworFullDataset(root, 'val.json', cfg_n, seq_len=16, stride=64, train=False)
        rng = np.random.default_rng(0)
        picks = rng.choice(len(ds_c), size=min(args.n, len(ds_c)), replace=False)

        sh, mc, mn, dc, dn = [], [], [], [], []
        with torch.no_grad():
            for i in picks:
                out = {}
                for tag, ds, s in (('c', ds_c, s_c), ('n', ds_n, s_n)):
                    b = ds[int(i)]
                    bb = {k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else v)
                          for k, v in b.items()}
                    o = model.forward_step(bb, train=False)
                    out[tag] = (o['pred_keypoints_3d'].cpu(), o['pred_keypoints_2d'].cpu(),
                                b['gt_j3d_wo_trans'], b['gt_j2d'], b['gt_j2d_conf'],
                                b['gt_valid'], b['img_size'][0], s)
                v = out['c'][5].reshape(-1).bool()
                if not v.any():
                    continue
                p3c, p3n = out['c'][0].flatten(0, 1)[v], out['n'][0].flatten(0, 1)[v]
                sh.append((p3n - p3c).norm(dim=-1).flatten() * 1000)
                for tag, acc2, acc3 in (('c', dc, mc), ('n', dn, mn)):
                    p3, p2, g3, g2, c2, _, size, s = out[tag]
                    g3 = g3.flatten(0, 1)[v]
                    acc3.append(wrist_relative(p3.flatten(0, 1)[v], g3).flatten())
                    # dn * out / s -> source pixels, so both runs share a unit
                    e = (p2.flatten(0, 1)[v] - g2.flatten(0, 1)[v]) * size / s
                    w = c2.flatten(0, 1)[v] > 0
                    if w.any():
                        acc2.append(e.norm(dim=-1)[w])
        if not sh:
            print(f'{name:8} no valid hands')
            continue
        f = lambda L: torch.cat(L).mean().item()
        gc = (768 // 16) * (1024 // 16)
        on = (-(-int(round(H0 * s_n)) // 16) * 16, -(-int(round(W0 * s_n)) // 16) * 16)
        gn = (on[0] // 16) * (on[1] // 16)
        rows.append((name, f(sh), f(mc), f(mn), f(dc), f(dn), gc, gn))
        print(f'{name:8} {f"{W0}x{H0}":>10} {f"{gc}tok":>10} {f"{gn}tok":>10} '
              f'{f(sh):6.2f} mm {f(mc):10.2f} mm {f(mn):10.2f} mm '
              f'{f(dc):7.1f} px {f(dn):7.1f} px')

    if rows:
        print('-' * 108)
        w = np.array([.221, .394, .178, .085, .062, .060])   # epoch share, DATASETS order
        keep = np.array([r[0] for r in rows])
        idx = [['hot3d', 'dexycb', 'arctic', 'ho3d', 'h2o', 'h2o3d'].index(k) for k in keep]
        w = w[idx] / w[idx].sum()
        agg = lambda j: float(sum(wi * r[j] for wi, r in zip(w, rows)))
        print(f'{"weighted":8} {"":>10} {"":>10} {"":>10} {agg(1):6.2f} mm '
              f'{agg(2):10.2f} mm {agg(3):10.2f} mm {agg(4):7.1f} px {agg(5):7.1f} px')
        print(f'\nMPJPE delta (native - canvas): {agg(3) - agg(2):+.2f} mm weighted.')
    print('\nCompare shift: mid-trunk pooling 15.73mm, uniform 640x480 5.70mm,')
    print('hand-token masking 20.96mm -- all on this same checkpoint.')


if __name__ == '__main__':
    raise SystemExit(main())
