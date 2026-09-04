"""
Convert DexYCB into the export layout that
lib/datasets/hawor_preprocess_train.py consumes.

DexYCB is a seated single-hand grasping capture: 10 subjects, 8 static
RealSense cameras, ~1000 sequences. Three things need care.

1. Pose representation. `pose_m` is [1, 51]: axis-angle global orientation in
   `[0:3]`, **45 MANO PCA coefficients** in `[3:48]`, and translation in
   `[48:51]`. dex-ycb-toolkit builds `ManoLayer(flat_hand_mean=False,
   ncomps=45, use_pca=True)`, so recovering the axis-angle pose that HaWoR's
   run_mano wants (it is called with rotation matrices, pose2rot=False) needs
   BOTH steps:

       axis_angle_45 = hands_mean + pca_coeffs @ hands_components[:45]

   Dropping the PCA projection is obvious. Dropping `hands_mean` is not: it is
   the same silent failure that displaced ARCTIC by ~75mm and survived a 2D
   overlay check. See mano_pca_to_axis_angle().

2. Camera. Intrinsics are per-serial with fx != fy and an off-centre principal
   point (e.g. ppx=302.4 on a 640-wide image, 17.6px from centre). HaWoR
   assumes one focal length with the principal point at the image centre, so
   frames are translated to put it there. There is no distortion to undo --
   the released colour images are already rectified.

3. Handedness. Sessions are split 50/50 between left and right hands (200 each
   over the first four subjects), and `meta.yml: mano_sides` says which -- NOT
   the mano_calib name. Every calibration directory is named `*_right` because
   DexYCB fits one shape per subject from the right hand and reuses it for
   both, so reading handedness off those names gives "all right" and is wrong
   for half the data.

   Only one hand is ever in shot, so the other is genuinely absent rather than
   unlabelled; hawor_preprocess_train derives validity from `any(rot != 0)`,
   so leaving that side's arrays zero is the correct label.

   The PCA basis is side-specific: MANO_LEFT.pkl carries its own hands_mean and
   hands_components, and using the right basis for a left hand is exactly what
   the joint check exists to catch.

Every sequence is verified against DexYCB's own `joint_3d` before it is
written; a mismatch raises rather than emitting quietly wrong data.

Usage:
    python lib/datasets/dexycb_to_export.py \
        --dexycb_root datasets/dexycb --out_root datasets/dexycb_export \
        --workers 8 --split train
"""
import argparse
import os
import pickle
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

sys.path.append(os.path.abspath('.'))

import cv2
import joblib
import numpy as np
import torch
import yaml

# load_gt_cam() computes head_pose @ ego_extrinsics @ R_90. DexYCB cameras are
# static, so we define the world frame to BE the camera frame and cancel R_90,
# which leaves camera-to-world as the identity.
R_90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

# DexYCB `joint_3d` follows manopth's order (dex_ycb.py:_MANO_JOINTS): wrist,
# then thumb/index/middle/ring/little as mcp,pip,dip,tip. HaWoR's MANO wrapper
# already applies `mano_to_openpose` (lib/models/mano_wrapper.py:26), which
# produces that same order -- so no permutation is needed here. Measured: the
# 16 kinematic joints agree to 0.00007mm.
#
# The 5 fingertips are excluded from the check. They disagree by 1.5-5mm
# because smplx picks its tip vertices from `vertex_ids['mano']` while manopth
# uses a different set; that is a definitional difference about which mesh
# vertex counts as the tip, not a pose error. Only the 16 regressed joints are
# a shared, well-defined quantity.
NON_TIP_JOINTS = [i for i in range(21) if i % 4 != 0 or i == 0]
_MANO_CACHE = {}


def mano_pca_basis(side='right'):
    """(hands_mean (45,), hands_components (45,45)) for one MANO side."""
    if side not in _MANO_CACHE:
        pkl = ('_DATA/data/mano/MANO_RIGHT.pkl' if side == 'right'
               else '_DATA/data_left/mano_left/MANO_LEFT.pkl')
        with open(pkl, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        _MANO_CACHE[side] = (np.array(d['hands_mean'], dtype=np.float64).reshape(45),
                             np.array(d['hands_components'], dtype=np.float64)[:45])
    return _MANO_CACHE[side]


def mano_pca_to_axis_angle(pca, side='right'):
    """(...,45) PCA coefficients -> (...,45) axis-angle, matching manopth's
    ManoLayer(flat_hand_mean=False, ncomps=45, use_pca=True)."""
    mean, comps = mano_pca_basis(side)
    return pca @ comps + mean


class _DexYCBLoader(yaml.SafeLoader):
    """SafeLoader that understands the one non-standard tag DexYCB emits.

    calibration/intrinsics/*.yml stores the stereo extrinsics as a
    `!!python/tuple`, which SafeLoader refuses. yaml.unsafe_load would parse it
    but would also execute any other python tag in the file, so the tag is
    registered explicitly instead.
    """


_DexYCBLoader.add_constructor(
    'tag:yaml.org,2002:python/tuple',
    lambda loader, node: tuple(loader.construct_sequence(node)))


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.load(f, Loader=_DexYCBLoader)


def list_sequences(root):
    """(subject, session, seq_dir) for every sequence with a meta.yml."""
    out = []
    for sub in sorted(d for d in os.listdir(root) if d.endswith(tuple(f'subject-{i:02d}' for i in range(1, 11)))):
        for sess in sorted(os.listdir(os.path.join(root, sub))):
            d = os.path.join(root, sub, sess)
            if os.path.isfile(os.path.join(d, 'meta.yml')):
                out.append((sub, sess, d))
    return out


def convert_sequence(seq_dir, calib_root, serial, out_dir, jpeg_q=92, tol_mm=0.01):
    meta = load_yaml(os.path.join(seq_dir, 'meta.yml'))
    if serial not in meta['serials']:
        raise RuntimeError(f'{seq_dir}: camera {serial} not in this sequence')
    side = meta['mano_sides'][0]
    if side not in ('left', 'right'):
        raise RuntimeError(f'{seq_dir}: unexpected hand side {side!r}')
    suf = "l" if side == "left" else "r"

    intr = load_yaml(os.path.join(calib_root, 'intrinsics', f'{serial}_640x480.yml'))['color']
    betas = np.asarray(load_yaml(os.path.join(
        calib_root, f'mano_{meta["mano_calib"][0]}', 'mano.yml'))['betas'], dtype=np.float32)

    cam_dir = os.path.join(seq_dir, serial)
    T = int(meta['num_frames'])
    os.makedirs(os.path.join(out_dir, 'extracted_images'), exist_ok=True)

    # fx and fy differ by <0.1% on these cameras; HaWoR carries a single focal.
    focal = float((intr['fx'] + intr['fy']) / 2.0)
    ppx, ppy = float(intr['ppx']), float(intr['ppy'])

    anno = {f'{k}_{s}': np.zeros((T, d), dtype=np.float32)
            for k, d in [('rot', 3), ('trans', 3), ('pose', 45), ('betas', 10)]
            for s in ('l', 'r')}

    check_pred, check_gt = [], []
    W = H = None
    n_valid = 0
    for t in range(T):
        img_p = os.path.join(cam_dir, f'color_{t:06d}.jpg')
        lab_p = os.path.join(cam_dir, f'labels_{t:06d}.npz')
        if not (os.path.exists(img_p) and os.path.exists(lab_p)):
            raise RuntimeError(f'{cam_dir}: missing frame {t}')
        img = cv2.imread(img_p)
        if W is None:
            H, W = img.shape[:2]
        # Translate so the principal point lands on the image centre.
        M = np.float32([[1, 0, W / 2.0 - ppx], [0, 1, H / 2.0 - ppy]])
        cv2.imwrite(os.path.join(out_dir, 'extracted_images', f'{t:04d}.jpg'),
                    cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

        z = np.load(lab_p)
        pm = z['pose_m'][0].astype(np.float64)
        if not np.any(pm):
            continue                       # no hand annotation for this frame
        anno[f'rot_{suf}'][t] = pm[0:3]
        anno[f'pose_{suf}'][t] = mano_pca_to_axis_angle(pm[3:48], side)
        anno[f'trans_{suf}'][t] = pm[48:51]
        anno[f'betas_{suf}'][t] = betas
        n_valid += 1

        j = z['joint_3d'][0]
        if not np.allclose(j, -1):
            check_pred.append(t)
            check_gt.append(j)

    if n_valid == 0:
        raise RuntimeError(f'{cam_dir}: no annotated frames')

    # --- verify against DexYCB's own joint_3d -------------------------------
    # This is the gate that a wrong PCA basis, a missing hands_mean or a bad
    # joint permutation must not get past.
    if check_pred:
        from hawor.utils.process import run_mano, run_mano_left
        fn = run_mano if side == 'right' else run_mano_left
        # Check in DexYCB's own convention (see the marker written below), so
        # the gate tests our conversion rather than the shapedirs disagreement.
        kw = {} if side == 'right' else {'fix_shapedirs': False}
        idx = np.asarray(check_pred)
        out = fn(torch.from_numpy(anno[f'trans_{suf}'][idx]).float()[None],
                 torch.from_numpy(anno[f'rot_{suf}'][idx]).float()[None],
                 torch.from_numpy(anno[f'pose_{suf}'][idx]).float()[None].reshape(1, len(idx), 45),
                 betas=torch.from_numpy(anno[f'betas_{suf}'][idx]).float()[None],
                 use_cuda=False, **kw)
        pred = out['joints'][0].numpy()[:, :21][:, NON_TIP_JOINTS]
        err = np.linalg.norm(pred - np.asarray(check_gt)[:, NON_TIP_JOINTS],
                             axis=-1) * 1000.0
        if err.mean() > tol_mm:
            raise RuntimeError(
                f'{cam_dir}: joint check FAILED, mean {err.mean():.5f}mm '
                f'max {err.max():.5f}mm over {len(idx)} frames (tol {tol_mm}mm). '
                f'Refusing to write -- suspect the PCA basis or hands_mean.')
        j_err = float(err.mean())
    else:
        j_err = float('nan')

    if side == 'left':
        # DexYCB fits betas with manopth, whose left MANO keeps the mirrored
        # shapedirs of smplx issue #48. hawor_preprocess_train reads this marker
        # and turns its correction off for this sequence; without it the shape
        # is reinterpreted and the joints move ~15mm.
        open(os.path.join(out_dir, 'mano_left_unfixed'), 'w').close()

    head_pose = np.tile(np.linalg.inv(R_90).astype(np.float32), (T, 1, 1))
    with open(os.path.join(out_dir, 'head_pose.pkl'), 'wb') as f:
        pickle.dump(head_pose, f)
    with open(os.path.join(out_dir, 'ego_extrinsics.pkl'), 'wb') as f:
        pickle.dump(np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)), f)
    with open(os.path.join(out_dir, 'focal.txt'), 'w') as f:
        f.write(str(focal))
    joblib.dump({k: torch.from_numpy(v) for k, v in anno.items()},
                os.path.join(out_dir, 'anno.pth'))
    return T, n_valid, j_err


def _worker(args):
    seq_dir, calib_root, serial, out_dir = args
    name = os.path.basename(out_dir)
    try:
        if os.path.exists(os.path.join(out_dir, 'anno.pth')):
            return name, 'skip', 0, 0, float('nan')
        T, n, e = convert_sequence(seq_dir, calib_root, serial, out_dir)
        return name, 'ok', T, n, e
    except Exception:
        traceback.print_exc()
        return name, 'fail', 0, 0, float('nan')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dexycb_root', required=True,
                   help='Directory holding 2020*-subject-* and calibration/')
    p.add_argument('--out_root', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--split', default='train', help='Manifest name (<split>.json)')
    p.add_argument('--subjects', nargs='*', default=None,
                   help='Restrict to these subject dir names (held-out splits)')
    p.add_argument('--cameras', type=int, default=1,
                   help='How many of the 8 views to export per sequence. The '
                        'views share a hand pose, so extra views add little '
                        'diversity for a lot of frames.')
    p.add_argument('--serials', nargs='*', default=None,
                   help='Export exactly these camera serials instead of the '
                        'first --cameras of them. The 8 views differ a lot in '
                        'character (overhead vs a side view showing the whole '
                        'subject), so which one you take is not incidental.')
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    calib_root = os.path.join(args.dexycb_root, 'calibration')
    if not os.path.isdir(calib_root):
        raise SystemExit(f'{calib_root} not found; extract calibration.tar.gz first')

    seqs = list_sequences(args.dexycb_root)
    if args.subjects:
        seqs = [s for s in seqs if s[0] in args.subjects]
    if args.limit:
        seqs = seqs[:args.limit]
    if not seqs:
        raise SystemExit(f'No sequences under {args.dexycb_root}')

    jobs = []
    for sub, sess, d in seqs:
        avail = load_yaml(os.path.join(d, 'meta.yml'))['serials']
        if args.serials:
            serials = [x for x in args.serials if x in avail]
            if not serials:
                print(f'  WARNING: none of --serials present in {sub}/{sess}')
        else:
            serials = avail[:args.cameras]
        for serial in serials:
            name = f'{sub}__{sess}__{serial}'
            jobs.append((d, calib_root, serial, os.path.join(args.out_root, name)))
    os.makedirs(args.out_root, exist_ok=True)
    print(f'{len(seqs)} sequences x {args.cameras} view(s) = {len(jobs)} outputs')

    done, failed, frames, errs = [], [], 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            name, status, T, n, e = fut.result()
            (failed if status == 'fail' else done).append(name)
            frames += n
            if e == e:
                errs.append(e)
            if k % 20 == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} ({len(failed)} failed)', flush=True)

    manifest = os.path.join(args.out_root, f'{args.split}.json')
    import json
    with open(manifest, 'w') as f:
        json.dump(sorted(done), f, indent=1)
    print(f'converted {len(done)} outputs ({frames} annotated frames), '
          f'{len(failed)} failed')
    if errs:
        print(f'joint check vs DexYCB joint_3d: mean {np.mean(errs):.4f}mm, '
              f'worst sequence {np.max(errs):.4f}mm')
    print(f'wrote {manifest}')


if __name__ == '__main__':
    main()
