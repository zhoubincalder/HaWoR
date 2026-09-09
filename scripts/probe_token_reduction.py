"""
How far does a token-reduction scheme move a TRAINED model's predictions?

Neither pooling nor token merging can be evaluated by a throughput number, and
training each variant to convergence costs days. This measures the perturbation
instead: run the same real batches through a trained checkpoint twice, once
unmodified and once with tokens reduced, and report how far the predicted joints
move.

That is an UPPER BOUND on the accuracy cost, not the cost itself -- fine-tuning
would recover much of it, since the head can re-adapt to whatever the trunk now
emits. But it separates the two things that get conflated:

  * a scheme that barely moves a trained model's output is a small perturbation
    the fine-tune will absorb;
  * one that moves it tens of millimetres is a different representation, and the
    head has to relearn placement from scratch.

Reported in millimetres of 3D joint displacement and pixels of 2D displacement,
against the model's own unmodified predictions -- not against ground truth, so
this isolates the effect of the reduction from the model's own error.
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath('.'))

import numpy as np
import torch


def load(cfg_path, ckpt, **overrides):
    from hawor.configs import get_config
    from lib.models.hawor_full import HaworFull
    cfg = get_config(cfg_path, merge=True, update_cachedir=True)
    cfg.defrost()
    cfg.MODEL.INPUT_H = overrides.pop('h', 1024)
    cfg.MODEL.INPUT_W = overrides.pop('w', 768)
    cfg.MODEL.BACKBONE.FREEZE = True          # eval only
    cfg.MODEL.BACKBONE.GRAD_CHECKPOINT = False
    cfg.MODEL.BACKBONE.TORCH_COMPILE = 0
    cfg.MODEL.BACKBONE.FP8 = False
    cfg.MODEL.BACKBONE.FP8_TRAINING = False
    pool_after = overrides.pop('pool_after', 0)
    cfg.MODEL.WARM_START = ''
    cfg.freeze()
    model = HaworFull(cfg)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    nb_missing = [k for k in missing if not k.startswith('backbone.')]
    if nb_missing:
        print(f'  WARNING {len(nb_missing)} non-backbone keys missing: {nb_missing[:4]}')
    if pool_after:
        model.backbone.enable_midtrunk_pool(pool_after)
    return model.cuda().eval(), cfg


def batches(cfg, root, n, seq_len=16):
    from lib.datasets.hawor_full_dataset import HaworFullDataset
    ds = HaworFullDataset(root, 'val.json', cfg, seq_len=seq_len, stride=64, train=False)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)
    for i in idx:
        b = ds[int(i)]
        yield {k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else v)
               for k, v in b.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--cfg', default='hawor/configs/hawor_full_sapiens2_1024.yaml')
    ap.add_argument('--root', default='datasets/hot3d_clips_export')
    ap.add_argument('--n', type=int, default=12, help='val windows to compare')
    ap.add_argument('--pool_after', type=int, default=12)
    ap.add_argument('--low_res', nargs=2, type=int, default=None,
                    metavar=('H', 'W'), help='also compare a lower input size')
    args = ap.parse_args()

    print(f'reference: unmodified trunk, 1024x768, {args.ckpt}')
    ref, cfg = load(args.cfg, args.ckpt)

    variants = [(f'pool 2x2 after block {args.pool_after}',
                 dict(pool_after=args.pool_after))]
    if args.low_res:
        variants.append((f'input {args.low_res[0]}x{args.low_res[1]}',
                         dict(h=args.low_res[0], w=args.low_res[1])))

    # Reference predictions first, so every variant is compared to the same run.
    ref_out = []
    with torch.no_grad():
        for b in batches(cfg, args.root, args.n):
            o = ref.forward_step(b, train=False)
            ref_out.append((o['pred_keypoints_3d'].cpu(),
                            o['pred_keypoints_2d'].cpu(),
                            b['gt_valid'].cpu()))
    del ref
    torch.cuda.empty_cache()

    print(f'\n{"variant":34} {"3D shift":>12} {"2D shift":>14} {"wrist depth":>13}')
    print('-' * 76)
    for name, ov in variants:
        m, c = load(args.cfg, args.ckpt, **ov)
        d3, d2, dz = [], [], []
        with torch.no_grad():
            for (r3, r2, valid), b in zip(ref_out, batches(c, args.root, args.n)):
                o = m.forward_step(b, train=False)
                v = valid.flatten(0, 1).bool()          # (B*T,2)
                p3 = o['pred_keypoints_3d'].cpu()
                p2 = o['pred_keypoints_2d'].cpu()
                if not v.any():
                    continue
                # 3D: displacement of every joint, in mm
                d3.append((p3[v] - r3[v]).norm(dim=-1).flatten() * 1000)
                # 2D: gt_j2d is normalized over the input frame, so scale back
                px = torch.tensor([c.MODEL.INPUT_W, c.MODEL.INPUT_H])
                d2.append(((p2[v] - r2[v]) * px).norm(dim=-1).flatten())
                # wrist depth specifically: the translation decode's own output
                dz.append((p3[v][:, 0, 2] - r3[v][:, 0, 2]).abs() * 1000)
        if not d3:
            print(f'{name:34} {"no valid hands":>12}')
            continue
        a3 = torch.cat(d3); a2 = torch.cat(d2); az = torch.cat(dz)
        print(f'{name:34} {a3.mean():8.2f} mm {a2.mean():10.2f} px '
              f'{az.mean():9.2f} mm')
        print(f'{"":34} {"median " + f"{a3.median():.2f}":>12} '
              f'{"median " + f"{a2.median():.2f}":>14}')
        del m
        torch.cuda.empty_cache()

    print('\nUpper bound: a fine-tune re-adapts the head, so the trained-from-here')
    print('cost is lower than this. Zero displacement would mean the reduction is')
    print('invisible to the trained model; tens of mm means a new representation.')


if __name__ == '__main__':
    raise SystemExit(main())
