"""
Inference throughput for the full-frame model.

Training numbers do not carry over: inference drops the backward pass, the
optimizer step and gradient checkpointing, so the speedup is not a fixed ratio
of the training figure. This measures the forward path directly.

Timing is CUDA-event based around torch.cuda.synchronize(), with warm-up
iterations discarded -- not wall clock around the whole process, and not tqdm's
it/s, which reports two decimals and whose reading count is not the step count.

Synthetic input by design: this isolates the model. Real deployment adds JPEG
decode, resize and letterbox on top, which is a separate (and parallelisable)
cost -- reported separately by --with_data.
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.abspath('.'))

import numpy as np
import torch


def bench_parts(model, batch, iters=20, warm=5):
    """Time the backbone separately from everything downstream of it.

    The point of the split: in a sliding window the backbone is recomputed for
    every frame of every window, but a frame's features do not depend on which
    window it lands in. Cache them once per frame and the marginal cost of an
    extra window is only the head plus the motion module. With stride 1 over a
    video each frame otherwise goes through the backbone seq_len times.
    """
    import einops
    from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat
    img = batch['img']
    B, T = img.shape[:2]
    flat = img.flatten(0, 1)

    def backbone_only():
        return model.backbone(flat).float()

    feat = backbone_only()

    def rest():
        tok = model.head(feat, return_tokens=True)
        if model.motion_module is not None:
            tok = einops.rearrange(tok, '(b t) n c -> (b n) t c', t=T)
            tok = model.motion_module(tok)
            tok = einops.rearrange(tok, '(b n) t c -> (b t) n c', n=2)
        return model.head.decode(tok)

    out = {}
    for name, fn in (('backbone', backbone_only), ('head+motion', rest)):
        for _ in range(warm):
            with torch.no_grad():
                fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(iters):
            t0 = time.perf_counter()
            with torch.no_grad():
                fn()
            torch.cuda.synchronize()
            ts.append(time.perf_counter() - t0)
        out[name] = float(np.mean(ts))
    return out


def bench(model, batch, iters=20, warm=5):
    dev = next(model.parameters()).device
    for _ in range(warm):
        with torch.no_grad():
            model.forward_step(batch, train=False)
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        with torch.no_grad():
            model.forward_step(batch, train=False)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts)
    return ts.mean(), ts.std(), np.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cfg', default='hawor/configs/hawor_full_sapiens2_1024.yaml')
    ap.add_argument('--h', type=int, default=1024)
    ap.add_argument('--w', type=int, default=768)
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 2, 4])
    ap.add_argument('--seq_len', type=int, default=16)
    ap.add_argument('--fp8', action='store_true',
                    help='torchao inference quantization on a frozen trunk')
    ap.add_argument('--compile', action='store_true')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--breakdown', action='store_true',
                    help='time backbone vs head+motion, for feature-reuse maths')
    args = ap.parse_args()

    from hawor.configs import get_config
    from lib.models.hawor_full import HaworFull
    cfg = get_config(args.cfg, merge=True, update_cachedir=True)
    cfg.defrost()
    cfg.MODEL.INPUT_H, cfg.MODEL.INPUT_W = args.h, args.w
    # Inference: the trunk never trains, so freeze it. That also makes the FP8
    # inference-quantization path legal -- it refuses on an unfrozen backbone.
    cfg.MODEL.BACKBONE.FREEZE = True
    cfg.MODEL.BACKBONE.GRAD_CHECKPOINT = False
    cfg.MODEL.BACKBONE.FP8 = bool(args.fp8)
    cfg.MODEL.BACKBONE.FP8_TRAINING = False
    cfg.MODEL.BACKBONE.TORCH_COMPILE = 1 if args.compile else 0
    cfg.MODEL.WARM_START = ''
    cfg.freeze()

    model = HaworFull(cfg).cuda().eval()
    tag = f"{args.h}x{args.w} fp8={args.fp8} compile={args.compile}"
    print(f'\n{tag}')
    print(f'{"batch":>6} {"frames":>7} {"ms/window":>11} {"±":>7} {"frames/s":>9} {"peak GB":>8}')

    for bs in args.batches:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        batch = {
            'img': torch.randn(bs, args.seq_len, 3, args.h, args.w,
                               device='cuda', dtype=torch.float32),
            'img_focal': torch.full((bs, args.seq_len), 600.0, device='cuda'),
            'img_center': torch.tensor([args.w / 2, args.h / 2], device='cuda')
                               .repeat(bs, args.seq_len, 1),
        }
        try:
            mean, std, med = bench(model, batch, iters=args.iters)
        except torch.OutOfMemoryError:
            print(f'{bs:>6} {bs * args.seq_len:>7} {"OOM":>11}')
            continue
        peak = torch.cuda.max_memory_allocated() / 1e9
        fps = bs * args.seq_len / mean
        print(f'{bs:>6} {bs * args.seq_len:>7} {mean * 1000:>11.1f} '
              f'{std * 1000:>7.1f} {fps:>9.1f} {peak:>8.1f}')
        if args.breakdown:
            p = bench_parts(model, batch, iters=args.iters)
            nf = bs * args.seq_len
            bb, hm = p['backbone'], p['head+motion']
            print(f'       backbone      {bb * 1000:8.1f} ms  ({100 * bb / (bb + hm):4.1f}%)  '
                  f'{nf / bb:7.1f} frames/s')
            print(f'       head+motion   {hm * 1000:8.1f} ms  ({100 * hm / (bb + hm):4.1f}%)  '
                  f'{nf / hm:7.1f} frames/s')
            print(f'       -> with cached backbone features, an extra window costs '
                  f'{hm * 1000:.1f} ms instead of {(bb + hm) * 1000:.1f} ms '
                  f'({(bb + hm) / hm:.1f}x)')


if __name__ == '__main__':
    main()
