"""
Convert H2O-3D v1 into the export layout that
lib/datasets/hawor_preprocess_train.py consumes.

H2O-3D is the two-handed companion to HO-3D (same authors, same HOnnotate
annotation pipeline): 76,340 frames of two hands manipulating a YCB object.
Unlike HO-3D and DexYCB it labels BOTH hands per frame, which is what the
full-frame two-hand model actually wants.

1. Pose encoding and joint order. Both are the OPPOSITE of DexYCB's, so the
   two converters must not be assumed to share logic:

                        DexYCB                      H2O-3D
     pose encoding      45 PCA coefficients         raw axis-angle
     joint order        OpenPose (identity)         MANO order (needs M2S)

   Treating the pose as PCA costs 7.5mm (right) / 23.6mm (left); leaving the
   stored joints in MANO order costs ~60mm. Neither is large enough to be
   obvious in a 2D overlay -- which is how the equivalent ARCTIC mistake
   survived review -- so every sequence is checked against the dataset's own
   joints and the converter refuses to write on a mismatch.

2. Camera. camMat is constant per sequence with an off-centre principal point
   (cx=315.73 on a 640-wide image). HaWoR assumes it at the image centre, so
   frames are translated to put it there.

3. Coordinate frame. H2O-3D inherits HO-3D's OpenGL convention: the camera
   looks down -z, so stored joints have negative z (e.g. -0.71..-0.56 m) and
   HaWoR's projection, which assumes +z forward, drops every frame as
   off-image. The fix is a change of basis by diag(1,-1,-1).

   That matrix has determinant +1, so it is a proper rotation (pi about x) and
   folds into the global orientation -- but MANO rotates the hand about its
   REST-POSE ROOT JOINT J0, not the origin, so the translation must be
   re-anchored too:

       R_global' = F @ R_global
       trans'    = F @ (J0 + trans) - J0

   Composing F without re-anchoring leaves a 18.4mm error, which is |2*(J0_y,
   J0_z)| -- small enough to look like noise and large enough to matter.

   Note this is precisely what the first version of this converter got wrong,
   and note *how* it was caught: the joint check below passed at 0.00005mm
   while the data was still unusable, because it compared MANO output against
   stored joints that were both in H2O-3D's frame. Self-consistency is not
   correctness. The frame error only surfaced when hawor_preprocess_train
   reported 0 valid frames of 699. The check now runs against joints
   transformed into OpenCV coordinates, so it tests the frame as well.

Usage:
    python lib/datasets/h2o3d_to_export.py \
        --h2o3d_root datasets/h2o3d --out_root datasets/h2o3d_export \
        --split train --workers 8
"""
import argparse
import json
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

# load_gt_cam() computes head_pose @ ego_extrinsics @ R_90. H2O-3D cameras are
# static, so the world frame is defined to BE the camera frame and R_90 is
# cancelled, leaving camera-to-world as the identity.
R_90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

# HO3D/H2O-3D store joints in MANO's own order; HaWoR's MANO wrapper emits the
# OpenPose order (lib/models/mano_wrapper.py:26 applies mano_to_openpose). This
# is HO3D's published `jointsMapManoToSimple`, used to bring the *stored*
# joints into the order our MANO output already uses.
MANO_TO_SIMPLE = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]

# OpenGL (-z forward) -> OpenCV (+z forward). Proper rotation, det = +1.
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def _aa_to_R(a):
    return cv2.Rodrigues(np.asarray(a, dtype=np.float64))[0]


def _R_to_aa(R):
    return cv2.Rodrigues(R)[0].reshape(3)


_J0_CACHE = {}


def rest_root_joint(fn, betas):
    """Rest-pose root joint for these betas.

    MANO applies the global orientation about this point, so a change of basis
    has to be applied about it rather than about the origin."""
    key = (fn.__name__, betas.tobytes())
    if key not in _J0_CACHE:
        z3, z45 = torch.zeros(1, 1, 3), torch.zeros(1, 1, 45)
        out = fn(z3, z3, z45, betas=torch.from_numpy(betas).float()[None, None],
                 use_cuda=False)
        _J0_CACHE[key] = out['joints'][0, 0, 0].numpy().astype(np.float64)
    return _J0_CACHE[key]

# smplx and manopth disagree 1.5-5mm on which mesh vertex is a fingertip, so
# only the 16 regressed joints are a shared, well-defined quantity to check.
NON_TIP_JOINTS = [i for i in range(21) if i % 4 != 0 or i == 0]


def frame_ids(seq_dir):
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob(os.path.join(seq_dir, 'rgb', '*.jpg')))


def convert_sequence(seq_dir, out_dir, jpeg_q=92, tol_mm=0.01):
    ids = frame_ids(seq_dir)
    if not ids:
        raise RuntimeError(f'{seq_dir}: no rgb frames')
    T = len(ids)
    os.makedirs(os.path.join(out_dir, 'extracted_images'), exist_ok=True)

    anno = {f'{k}_{s}': np.zeros((T, d), dtype=np.float32)
            for k, d in [('rot', 3), ('trans', 3), ('pose', 45), ('betas', 10)]
            for s in ('l', 'r')}

    check = {'l': [], 'r': []}          # (frame index, stored joints)
    focal = ppx = ppy = None
    W = H = None
    n_valid = {'l': 0, 'r': 0}

    for i, fid in enumerate(ids):
        meta_p = os.path.join(seq_dir, 'meta', f'{fid}.pkl')
        if not os.path.exists(meta_p):
            raise RuntimeError(f'{seq_dir}: missing meta for frame {fid}')
        with open(meta_p, 'rb') as f:
            d = pickle.load(f, encoding='latin1')

        K = np.asarray(d['camMat'], dtype=np.float64)
        if focal is None:
            focal = float((K[0, 0] + K[1, 1]) / 2.0)   # fx and fy differ by <0.1%
            ppx, ppy = float(K[0, 2]), float(K[1, 2])
        elif abs(float(K[0, 0]) - K[0, 0]) > 1e-6 or abs(float(K[0, 2]) - ppx) > 1e-6:
            raise RuntimeError(f'{seq_dir}: camera intrinsics change mid-sequence')

        img = cv2.imread(os.path.join(seq_dir, 'rgb', f'{fid}.jpg'))
        if W is None:
            H, W = img.shape[:2]
        # Translate so the principal point lands on the image centre, which is
        # what HaWoR's projection assumes.
        M = np.float32([[1, 0, W / 2.0 - ppx], [0, 1, H / 2.0 - ppy]])
        cv2.imwrite(os.path.join(out_dir, 'extracted_images', f'{i:04d}.jpg'),
                    cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

        betas = np.asarray(d['handBeta'], dtype=np.float32)   # shared by both hands
        from hawor.utils.process import run_mano, run_mano_left
        for side, key, fn in (('l', 'left', run_mano_left), ('r', 'right', run_mano)):
            if not np.asarray(d[f'poseValid{key.capitalize()}']).all():
                continue                                      # hand absent -> rot stays 0
            p48 = np.asarray(d[f'{key}HandPose'], dtype=np.float64)
            tr = np.asarray(d[f'{key}HandTrans'], dtype=np.float64)
            J0 = rest_root_joint(fn, betas)
            # OpenGL -> OpenCV, applied about the rest-pose root joint.
            anno[f'rot_{side}'][i] = _R_to_aa(GL_TO_CV @ _aa_to_R(p48[:3]))
            anno[f'trans_{side}'][i] = GL_TO_CV @ (J0 + tr) - J0
            anno[f'pose_{side}'][i] = p48[3:48]                # already axis-angle
            anno[f'betas_{side}'][i] = betas
            n_valid[side] += 1
            j = np.asarray(d[f'{key}HandJoints3D'], dtype=np.float64)
            if np.isfinite(j).all():
                # compare in OpenCV coordinates, so the check tests the frame too
                check[side].append((i, (j @ GL_TO_CV.T)[MANO_TO_SIMPLE]))

    if n_valid['l'] == 0 and n_valid['r'] == 0:
        raise RuntimeError(f'{seq_dir}: no annotated frames')

    # --- verify against H2O-3D's own joints ---------------------------------
    # The gate that a wrong pose encoding or joint order must not get past.
    from hawor.utils.process import run_mano, run_mano_left
    errs = {}
    for side, fn in (('l', run_mano_left), ('r', run_mano)):
        if not check[side]:
            continue
        idx = np.array([c[0] for c in check[side]])
        gt = np.stack([c[1] for c in check[side]])
        out = fn(torch.from_numpy(anno[f'trans_{side}'][idx]).float()[None],
                 torch.from_numpy(anno[f'rot_{side}'][idx]).float()[None],
                 torch.from_numpy(anno[f'pose_{side}'][idx]).float()[None],
                 betas=torch.from_numpy(anno[f'betas_{side}'][idx]).float()[None],
                 use_cuda=False)
        pred = out['joints'][0].numpy()[:, :21]
        e = np.linalg.norm(pred[:, NON_TIP_JOINTS] - gt[:, NON_TIP_JOINTS], axis=-1) * 1000.0
        if e.mean() > tol_mm:
            raise RuntimeError(
                f'{seq_dir}: {side} joint check FAILED, mean {e.mean():.5f}mm '
                f'max {e.max():.5f}mm over {len(idx)} frames (tol {tol_mm}mm). '
                f'Refusing to write -- suspect the pose encoding (H2O-3D is '
                f'axis-angle, not PCA) or the MANO_TO_SIMPLE joint order.')
        errs[side] = float(e.mean())

    head_pose = np.tile(np.linalg.inv(R_90).astype(np.float32), (T, 1, 1))
    with open(os.path.join(out_dir, 'head_pose.pkl'), 'wb') as f:
        pickle.dump(head_pose, f)
    with open(os.path.join(out_dir, 'ego_extrinsics.pkl'), 'wb') as f:
        pickle.dump(np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)), f)
    with open(os.path.join(out_dir, 'focal.txt'), 'w') as f:
        f.write(str(focal))
    joblib.dump({k: torch.from_numpy(v) for k, v in anno.items()},
                os.path.join(out_dir, 'anno.pth'))
    return T, n_valid['l'], n_valid['r'], errs


def _worker(args):
    seq_dir, out_dir = args
    name = os.path.basename(out_dir)
    try:
        if os.path.exists(os.path.join(out_dir, 'anno.pth')):
            return name, 'skip', 0, 0, 0, {}
        T, nl, nr, e = convert_sequence(seq_dir, out_dir)
        return name, 'ok', T, nl, nr, e
    except Exception:
        traceback.print_exc()
        return name, 'fail', 0, 0, 0, {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h2o3d_root', required=True,
                   help='Directory holding train/ and evaluation/')
    p.add_argument('--out_root', required=True)
    p.add_argument('--source_split', default='train', choices=['train', 'evaluation'],
                   help='Which H2O-3D split to read. Its `evaluation` split has '
                        'no public hand labels in some releases -- check before '
                        'relying on it.')
    p.add_argument('--split', default='train', help='Manifest name (<split>.json)')
    p.add_argument('--sequences', nargs='*', default=None)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    src = os.path.join(args.h2o3d_root, args.source_split)
    seqs = sorted(d for d in os.listdir(src) if os.path.isdir(os.path.join(src, d)))
    if args.sequences:
        seqs = [s for s in seqs if s in args.sequences]
    if args.limit:
        seqs = seqs[:args.limit]
    if not seqs:
        raise SystemExit(f'No sequences under {src}')
    os.makedirs(args.out_root, exist_ok=True)
    print(f'{len(seqs)} sequences from {src}')

    jobs = [(os.path.join(src, s), os.path.join(args.out_root, s)) for s in seqs]
    done, failed, nl, nr, errs = [], [], 0, 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            name, status, T, l, r, e = fut.result()
            (failed if status == 'fail' else done).append(name)
            nl += l
            nr += r
            errs.extend(e.values())
            if k % 10 == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} ({len(failed)} failed)', flush=True)

    manifest = os.path.join(args.out_root, f'{args.split}.json')
    with open(manifest, 'w') as f:
        json.dump(sorted(done), f, indent=1)
    print(f'converted {len(done)} sequences, {len(failed)} failed')
    print(f'annotated hands: {nl} left / {nr} right')
    if errs:
        print(f'joint check vs H2O-3D joints: mean {np.mean(errs):.5f}mm, '
              f'worst {np.max(errs):.5f}mm')
    print(f'wrote {manifest}')


if __name__ == '__main__':
    main()
