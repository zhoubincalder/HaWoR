"""
How much does dropping non-hand patch tokens move a trained model's predictions?

The grid is 3072 tokens and two hands occupy 94-433 of them (3-14%), so most of
what the trunk computes is background. This measures the INFORMATION cost of
keeping only the hand neighbourhood, with the hand location taken from ground
truth so that localization error is excluded -- the question here is whether the
remaining tokens are sufficient, not whether we can find them.

MEASURED, and the answer is that dropping is WORSE than either alternative. On
the 0.8b full-frame checkpoint at 48x64:

    3072 (identity)  1.00x   0.00 mm    <- validity gate: the path is exact
    2816             1.09x   2.93 mm
    2048             1.50x  11.23 mm
    1536             2.00x  16.80 mm
    1024             3.00x  20.96 mm
     512             6.00x  16.52 mm
     256            12.00x  23.18 mm

Monotone to ~1536, then a ~20 +- 4 mm plateau: past half the grid the output is
decorrelated from the reference and the budget stops mattering. At a matched 3x
saving this is 20.96 mm against 5.70 mm for simply feeding 640x480, and it is
already past mid-trunk pooling's 15.73 mm by a 2x budget.

I predicted the opposite -- that dropping would beat pooling, since every
surviving token keeps its exact value and its true RoPE coordinates (cos/sin are
gathered with the same indices) while pooling invents token values by averaging
neighbours in a learned embedding space. Why that reasoning fails: preserving
the kept tokens' inputs does not preserve their outputs. Attention is global, so
every kept token's value changes once the key/value set shrinks, and
MANOTwoHandHead reads the grid as a permutation-invariant set with a softmax
over it (no positional embedding, see lib/models/modules.py) -- so removing
background redistributes attention mass rather than merely withholding
information. The trained model uses the background it was trained with.

Two further strikes against it as a lever: the hand locations here come from
ground truth, so a deployment would need a detector ahead of the trunk, which
gives up the single-pass end-to-end property; and the union mask over a 16-frame
window has to cover the hand's whole path, so the realizable budget is larger
than a per-frame bound suggests.

Note what this is NOT evidence of: Sapiens2's own 75%-mask pretraining
SUBSTITUTES a learned mask token and keeps all 3072 positions, so a shortened
sequence is not the pretraining regime. The in-distribution part is variable
sequence length (random_scale training) and per-token RoPE; the novel part is a
non-rectangular sparse subset.
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath('.'))

import numpy as np
import torch

HAWOR_HIDDEN = 1280


def load(cfg_path, ckpt, h=768, w=1024):
    from hawor.configs import get_config
    from lib.models.hawor_full import HaworFull
    cfg = get_config(cfg_path, merge=True, update_cachedir=True)
    cfg.defrost()
    cfg.MODEL.INPUT_H, cfg.MODEL.INPUT_W = h, w
    cfg.MODEL.BACKBONE.FREEZE = True
    cfg.MODEL.BACKBONE.GRAD_CHECKPOINT = False
    cfg.MODEL.BACKBONE.TORCH_COMPILE = 0
    cfg.MODEL.BACKBONE.FP8 = cfg.MODEL.BACKBONE.FP8_TRAINING = False
    cfg.MODEL.WARM_START = ''
    cfg.freeze()
    m = HaworFull(cfg)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    m.load_state_dict(sd.get('state_dict', sd), strict=False)
    return m.cuda().eval(), cfg


def hand_token_idx(gt_j2d, gt_valid, gh, gw, budget):
    """-> (budget,) patch indices covering where the hands go across the WINDOW.

    One index set for all 16 frames, not one per frame. That is required for
    correctness, not just convenience: RoPE cos/sin are indexed by patch and
    shared across the batch dimension, so per-frame index sets would give a
    token from frame t the position encoding of frame 0's token at the same rank
    -- systematically mismatched, and worse the further the hand has moved. An
    earlier version of this probe did exactly that and produced non-monotonic
    nonsense (33% of the grid scoring worse than 8%).

    A union mask is also what a real implementation would use: the hand travels
    within a window, so the kept region has to cover its whole path.
    """
    ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing='ij')
    centres = torch.stack([(xs.flatten() + 0.5) / gw - 0.5,
                           (ys.flatten() + 0.5) / gh - 0.5], -1)
    pts = [gt_j2d[t, slot] for t in range(gt_j2d.shape[0]) for slot in range(2)
           if gt_valid[t, slot] > 0]
    if pts:
        kp = torch.cat(pts, 0)
        d = (centres[:, None, :] - kp[None, :, :]).norm(dim=-1).min(1).values
    else:
        d = centres.norm(dim=-1)
    return torch.topk(d, budget, largest=False).indices.sort().values


def patch_masked_forward(wrapper, from_block):
    """Replace wrapper.forward with a pass that drops tokens outside _mask_idx."""
    bb = wrapper.backbone
    layers = wrapper._layers()

    def forward(x):
        dtype = next(bb.parameters()).dtype
        x = x.to(dtype)
        hs = bb.embeddings(x)
        cos, sin = bb.rope_embeddings(x)
        n_patch = cos.shape[0]
        n_prefix = hs.shape[1] - n_patch
        for layer in layers[:from_block]:
            hs = layer(hs, position_embeddings=(cos, sin))

        idx = wrapper._mask_idx.to(hs.device)                 # (K,) shared
        pre, pat = hs.split((n_prefix, n_patch), dim=1)
        b, k = pat.shape[0], idx.shape[0]
        pat = pat.index_select(1, idx)
        hs = torch.cat([pre, pat], dim=1)
        # Same indices into RoPE, so each kept token keeps its true position.
        pos = (cos.index_select(0, idx), sin.index_select(0, idx))
        for layer in layers[from_block:]:
            hs = layer(hs, position_embeddings=pos)
        if bb.config.normalize_backbone_outputs:
            hs = bb.norm(hs)
        feats = hs[:, n_prefix:, :].transpose(1, 2).reshape(b, -1, 1, k)
        return wrapper.proj(feats.to(wrapper.proj.weight.dtype))

    wrapper.forward = forward


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--cfg', default='hawor/configs/hawor_full_sapiens2_1024.yaml')
    ap.add_argument('--root', default='datasets/hot3d_clips_export')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--budgets', type=int, nargs='+', default=[1024, 512, 256])
    ap.add_argument('--from_block', type=int, default=0)
    args = ap.parse_args()

    from lib.datasets.hawor_full_dataset import HaworFullDataset
    ref, cfg = load(args.cfg, args.ckpt)
    gh = cfg.MODEL.INPUT_H // 16
    gw = cfg.MODEL.INPUT_W // 16
    print(f'grid {gh}x{gw} = {gh * gw} tokens, masking from block {args.from_block}')

    ds = HaworFullDataset(args.root, 'val.json', cfg, seq_len=16, stride=64, train=False)
    rng = np.random.default_rng(0)
    picks = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    samples, ref_out = [], []
    with torch.no_grad():
        for i in picks:
            b = ds[int(i)]
            bb = {k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else v)
                  for k, v in b.items()}
            o = ref.forward_step(bb, train=False)
            samples.append(b)
            ref_out.append((o['pred_keypoints_3d'].cpu(), bb['gt_valid'].cpu()))

    print(f'\n{"budget":>8} {"% of grid":>10} {"FLOP saving":>12} {"3D shift":>11}')
    print('-' * 46)
    for budget in args.budgets:
        m, c = load(args.cfg, args.ckpt)
        patch_masked_forward(m.backbone, args.from_block)
        shifts = []
        with torch.no_grad():
            for b, (r3, valid) in zip(samples, ref_out):
                idx = hand_token_idx(b['gt_j2d'], b['gt_valid'], gh, gw, budget)
                m.backbone._mask_idx = idx
                bb = {k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else v)
                      for k, v in b.items()}
                o = m.forward_step(bb, train=False)
                v = valid.flatten(0, 1).bool()
                if not v.any():
                    continue
                p3 = o['pred_keypoints_3d'].cpu()
                shifts.append((p3[v] - r3[v]).norm(dim=-1).flatten() * 1000)
        if not shifts:
            print(f'{budget:>8} {"no valid hands":>10}')
            continue
        s = torch.cat(shifts)
        frac = budget / (gh * gw)
        kept = args.from_block + (32 - args.from_block) * frac
        print(f'{budget:>8} {100 * frac:9.1f}% {32 / kept:11.2f}x '
              f'{s.mean():8.2f} mm')
        del m
        torch.cuda.empty_cache()

    print('\nCompare: mid-trunk 2x2 pooling moved predictions 15.73 mm for 1.85x,')
    print('and 640x480 input moved them 5.70 mm for ~3.0x.')


if __name__ == '__main__':
    raise SystemExit(main())
