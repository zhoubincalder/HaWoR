"""Evaluate a full-frame checkpoint on every dataset's val fold.

Reports the metrics a hand-pose result is normally judged on, per dataset and
corpus-weighted. What each one isolates:

  MPJPE      wrist-aligned per-joint error. Articulation only -- placement in
             the camera is removed by the alignment.
  PA-MPJPE   after a similarity transform (Procrustes). Shape and articulation
             with global rotation AND scale removed, so it is the kindest of
             the three and the one least sensitive to depth ambiguity.
  2D error   reprojection against the stored 2D labels, reported in SOURCE
             image pixels so datasets at different resolutions compare.
  root z     depth of the wrist. NOTE the annotations store only wrist-relative
             3D and 2D pixels -- there is no stored absolute translation -- so
             the reference here is DERIVED by least-squares fitting the GT 3D
             onto the GT 2D through the known intrinsics. It is a reconstruction
             of the ground truth, not the ground truth, and inherits whatever
             error that fit carries. Treat it as indicative.
  jitter     the project's own third-derivative measure (eval_utils.compute_jitter,
             fps^3 / 10), computed on predictions and on GT for reference. A
             model can be accurate per frame and still unusable if this is high.
  accel err  acceleration error against GT, mm/frame^2 -- temporal consistency
             without the fps^3 scaling.

Only frames with a valid hand contribute, and jitter/accel need runs of at
least 4 frames within a window, so they are computed per window per hand slot.
"""
import argparse
import os
import sys

sys.path.append(os.path.abspath('.'))

import numpy as np
import torch

DATASETS = [
    ('hot3d', 'datasets/hot3d_clips_export', 0.221),
    ('dexycb', 'datasets/dexycb_bronze_export', 0.394),
    ('arctic', 'datasets/arctic_export', 0.178),
    ('ho3d', 'datasets/ho3d_export', 0.085),
    ('h2o', 'datasets/h2o_export', 0.062),
    ('h2o3d', 'datasets/h2o3d_export', 0.060),
]


def procrustes(pred, gt):
    """Similarity-align pred onto gt (rotation, scale, translation). (N,J,3).

    Same construction as lib/eval_utils.compute_similarity_transform, written
    out here because that module imports matplotlib at import time and this
    environment has none. Batched over frames rather than looped per frame.
    """
    mu_p = pred.mean(1, keepdims=True)
    mu_g = gt.mean(1, keepdims=True)
    X, Y = pred - mu_p, gt - mu_g
    K = np.einsum('nji,njk->nik', X, Y)          # (N,3,3) cross-covariance
    U, sv, Vt = np.linalg.svd(K)
    Z = np.tile(np.eye(3), (pred.shape[0], 1, 1))
    # Reflection guard: a pure SVD solution can contain a reflection
    Z[:, 2, 2] = np.sign(np.linalg.det(np.einsum('nij,njk->nik', U, Vt)))
    R = np.einsum('nij,njk,nkl->nil', Vt.transpose(0, 2, 1), Z, U.transpose(0, 2, 1))
    var = (X ** 2).sum(axis=(1, 2))
    scale = np.einsum('nii->n', np.einsum('nij,njk->nik', Z, np.apply_along_axis(
        np.diag, 1, sv))) / np.maximum(var, 1e-12)
    aligned = scale[:, None, None] * np.einsum('nij,nkj->nki', R, X) + mu_g
    return aligned


def fit_root_depth(j3d_rel, j2d, focal, center):
    """Least-squares translation putting GT 3D onto GT 2D. -> (N,3).

    For each frame: find t minimising |proj(j3d_rel + t) - j2d|. Linearised
    exactly: for joint j with relative position p, the projection constraint
    (px - cx) * (p_z + t_z) = f * (p_x + t_x) is linear in t, so this is one
    least-squares solve per frame rather than an iterative PnP.
    """
    n, j = j3d_rel.shape[:2]
    out = np.full((n, 3), np.nan, np.float64)
    for i in range(n):
        p, q = j3d_rel[i], j2d[i]
        u = (q[:, 0] - center[0]) / focal
        v = (q[:, 1] - center[1]) / focal
        # u*(pz+tz) = px+tx  ->  tx - u*tz = u*pz - px
        A = np.zeros((2 * j, 3))
        b = np.zeros(2 * j)
        A[:j, 0] = 1.0; A[:j, 2] = -u; b[:j] = u * p[:, 2] - p[:, 0]
        A[j:, 1] = 1.0; A[j:, 2] = -v; b[j:] = v * p[:, 2] - p[:, 1]
        try:
            out[i] = np.linalg.lstsq(A, b, rcond=None)[0]
        except np.linalg.LinAlgError:
            pass
    return out


def jitter(j3d, fps=30.0):
    """eval_utils.compute_jitter, on (T,J,3) metres -> scalar (mm-scale/10)."""
    if j3d.shape[0] < 4:
        return np.nan
    d = j3d[3:] - 3 * j3d[2:-1] + 3 * j3d[1:-2] - j3d[:-3]
    return float(np.linalg.norm(d * fps ** 3, axis=2).mean() / 10.0)


def accel_err(pred, gt):
    """Mean acceleration error, mm/frame^2, on (T,J,3) metres."""
    if pred.shape[0] < 3:
        return np.nan
    ap = pred[2:] - 2 * pred[1:-1] + pred[:-2]
    ag = gt[2:] - 2 * gt[1:-1] + gt[:-2]
    return float(np.linalg.norm(ap - ag, axis=2).mean() * 1000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('ckpt')
    ap.add_argument('--cfg', default='hawor/configs/hawor_full_sapiens2_1024.yaml')
    ap.add_argument('--stride', type=int, default=256)
    ap.add_argument('--max_windows', type=int, default=60, help='per dataset')
    ap.add_argument('--only', nargs='*', default=None)
    args = ap.parse_args()

    from hawor.configs import get_config
    from lib.datasets.hawor_full_dataset import HaworFullDataset
    from lib.models.hawor_full import HaworFull

    cfg = get_config(args.cfg, merge=True, update_cachedir=True)
    cfg.defrost()
    cfg.MODEL.NATIVE_RES = True
    cfg.MODEL.MAX_TOKENS = 1550
    cfg.MODEL.RES_JITTER_LEVELS = []
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.MODEL.BACKBONE.FREEZE = True
    cfg.MODEL.BACKBONE.GRAD_CHECKPOINT = False
    cfg.MODEL.BACKBONE.TORCH_COMPILE = 0
    cfg.MODEL.BACKBONE.FP8 = cfg.MODEL.BACKBONE.FP8_TRAINING = False
    cfg.MODEL.WARM_START = ''
    cfg.freeze()

    model = HaworFull(cfg)
    sd = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    sd = sd.get('state_dict', sd)
    # A checkpoint saved while torch.compile was active carries an `_orig_mod.`
    # segment on every child of a compiled submodule -- 731 of 834 keys here.
    # Loading it into an uncompiled model matches only the 103 uncompiled ones
    # and silently leaves the rest at init, because this loads with strict=False.
    n_wrapped = sum(1 for k in sd if '_orig_mod.' in k)
    if n_wrapped:
        sd = {k.replace('._orig_mod.', '.'): v for k, v in sd.items()}
        print(f'stripped _orig_mod. from {n_wrapped} keys (checkpoint was compiled)')
    sd = {k: (v.float() if v.is_floating_point() else v) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f'loaded {len(sd) - len(unexpected)}/{len(sd)} tensors '
          f'({len(missing)} missing, {len(unexpected)} unexpected)')
    model = model.cuda().eval()

    rows = []
    for name, root, share in DATASETS:
        if (args.only and name not in args.only) or not os.path.isdir(root):
            continue
        ds = HaworFullDataset(root, 'val.json', cfg, seq_len=16,
                              stride=args.stride, train=False)
        idx = np.random.default_rng(0).choice(
            len(ds), size=min(args.max_windows, len(ds)), replace=False)
        mp, pa, e2d, ez, jp, jg, ae, nh = [], [], [], [], [], [], [], 0
        with torch.no_grad():
            for i in idx:
                b = ds[int(i)]
                bb = {k: (v.unsqueeze(0).cuda() if torch.is_tensor(v) else v)
                      for k, v in b.items()}
                o = model.forward_step(bb, train=False)
                # already (B*T,2,J,3) with B=1, so no batch axis to strip
                p3 = o['pred_keypoints_3d'].cpu().numpy()         # (T,2,J,3)
                p2 = o['pred_keypoints_2d'].cpu().numpy()
                g3 = b['gt_j3d_wo_trans'].numpy()
                g2 = b['gt_j2d'].numpy()
                gc = b['gt_j2d_conf'].numpy()
                val = b['gt_valid'].numpy() > 0                    # (T,2)
                W, H = (float(x) for x in b['img_size'][0])
                foc = float(b['img_focal'][0])
                ctr = b['img_center'][0].numpy()
                for slot in range(2):
                    m = val[:, slot]
                    if m.sum() < 1:
                        continue
                    nh += int(m.sum())
                    pr = p3[m, slot]
                    gt = g3[m, slot]
                    prw = pr - pr[:, :1]
                    gtw = gt - gt[:, :1]
                    mp.append(np.linalg.norm(prw - gtw, axis=-1).mean() * 1000)
                    pa.append(np.linalg.norm(procrustes(prw, gtw) - gtw, axis=-1).mean() * 1000)
                    # 2D: normalized -> input px -> source px
                    px = np.array([W, H])
                    d2 = (p2[m, slot] - g2[m, slot]) * px
                    w2 = gc[m, slot] > 0
                    if w2.any():
                        e2d.append(np.linalg.norm(d2, axis=-1)[w2].mean())
                    # root depth vs a GT fitted from the stored 2D + relative 3D
                    tg = fit_root_depth(gtw, g2[m, slot] * px, foc, ctr)
                    zp = pr[:, 0, 2]
                    ok = np.isfinite(tg[:, 2])
                    if ok.any():
                        ez.append(np.abs(zp[ok] - tg[ok, 2]).mean() * 1000)
                    if m.sum() >= 4:
                        jp.append(jitter(pr)); jg.append(jitter(gt + tg[:, None, :]))
                    if m.sum() >= 3:
                        ae.append(accel_err(prw, gtw))
        f = lambda a: float(np.nanmean(a)) if len(a) else float('nan')
        rows.append((name, share, len(idx), nh, f(mp), f(pa), f(e2d), f(ez), f(jp), f(jg), f(ae)))
        print(f'{name:8} n={nh:6}  MPJPE {f(mp):7.1f}  PA {f(pa):6.1f}  '
              f'2D {f(e2d):7.1f}px  rootZ {f(ez):8.1f}  jit {f(jp):7.1f}/{f(jg):.1f}  '
              f'accel {f(ae):6.2f}')

    if rows:
        w = np.array([r[1] for r in rows]); w = w / w.sum()
        print('\n' + '=' * 96)
        print(f'{"dataset":9}{"hands":>8}{"MPJPE":>9}{"PA-MPJPE":>10}{"2D px":>9}'
              f'{"rootZ mm":>10}{"jitter p/gt":>15}{"accel":>8}')
        print('-' * 96)
        for r in rows:
            print(f'{r[0]:9}{r[3]:8}{r[4]:9.1f}{r[5]:10.1f}{r[6]:9.1f}{r[7]:10.1f}'
                  f'{r[8]:8.1f}/{r[9]:<6.1f}{r[10]:8.2f}')
        print('-' * 96)
        agg = lambda j: float(sum(wi * r[j] for wi, r in zip(w, rows)))
        print(f'{"weighted":9}{sum(r[3] for r in rows):8}{agg(4):9.1f}{agg(5):10.1f}'
              f'{agg(6):9.1f}{agg(7):10.1f}{agg(8):8.1f}/{agg(9):<6.1f}{agg(10):8.2f}')
        print('\nmm unless noted. rootZ is against a DERIVED reference (see docstring).')


if __name__ == '__main__':
    raise SystemExit(main())
