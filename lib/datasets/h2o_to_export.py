"""
Convert H2O (Kwon et al., ICCV 2021) into the export layout that
lib/datasets/hawor_preprocess_train.py consumes.

H2O is the closest match in the corpus to what HaWoR stage A actually does:
egocentric, both hands labelled every frame, MANO parameters plus 3D
keypoints, and 1280x720 RGB -- higher resolution than HOT3D, ARCTIC, H2O-3D or
DexYCB, all of which are 640x480.

Not to be confused with H2O-3D (lib/datasets/h2o3d_to_export.py), which is a
different dataset by different authors with different conventions.

Conventions, each determined empirically against the shipped `hand_pose`
keypoints rather than assumed -- the four converters before this one disagreed
with each other on every one of these:

  pose encoding     raw axis-angle          (H2O-3D: same; DexYCB: 45 PCA)
  joint order       identity, OpenPose      (H2O-3D: needs jointsMapManoToSimple)
  coordinate frame  OpenCV, as-is           (H2O-3D: OpenGL, needs diag(1,-1,-1))
  left shapedirs    UNCORRECTED (manopth)   (DexYCB: same; H2O-3D/ARCTIC: corrected)

Measured on subject1/h1/0 frame 54: 0.00007mm right, 0.00003mm left.

That last row is why this writes the `mano_left_unfixed` marker. MANO_LEFT.pkl
ships mirrored shapedirs (smplx issue #48); run_mano_left corrects them by
default, but H2O's betas -- like DexYCB's -- were fitted against the
uncorrected model, so applying the correction moves the left hand 2.7mm.
hawor_preprocess_train reads the marker and disables the correction for these
sequences only.

Layout per sequence: <subject>_ego/<scene>/<seq>/cam4/{rgb,hand_pose_mano,
hand_pose,cam_pose}/ plus cam_intrinsics.txt. cam4 is the egocentric camera.

Splits come from the dataset's own label_split/pose_{train,val,test}.txt so
results stay comparable with published numbers.

Usage:
    python lib/datasets/h2o_to_export.py \
        --h2o_root datasets/h2o_ego --out_root datasets/h2o_export \
        --workers 8
"""
import argparse
import json
import os
import pickle
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from glob import glob

sys.path.append(os.path.abspath('.'))

import cv2
import joblib
import numpy as np
import torch

# load_gt_cam() computes head_pose @ ego_extrinsics @ R_90. H2O's hand labels
# are already in the egocentric camera frame, so the world frame is defined to
# BE that frame and R_90 is cancelled, leaving camera-to-world as the identity.
# (cam_pose/ holds the camera's pose in the scene, which stage A does not use --
# it is the SLAM stage's job to recover that.)
R_90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)

# smplx and manopth disagree 1.5-5mm on which mesh vertex is a fingertip, so
# only the 16 regressed joints are a shared, well-defined quantity to check.
NON_TIP_JOINTS = [i for i in range(21) if i % 4 != 0 or i == 0]

# hand_pose_mano: 2 hands x 62 = [valid(1), trans(3), pose(48), betas(10)]
# hand_pose:      2 hands x 64 = [valid(1), 21 joints x 3]
MANO_STRIDE, KP_STRIDE = 62, 64


def read_mano(path):
    """-> {'l'|'r': (valid, rot(3), pose(45), trans(3), betas(10))}"""
    v = np.loadtxt(path, dtype=np.float64)
    if v.shape[0] != 2 * MANO_STRIDE:
        raise RuntimeError(f'{path}: expected {2*MANO_STRIDE} values, got {v.shape[0]}')
    out = {}
    for side, off in (('l', 0), ('r', MANO_STRIDE)):
        h = v[off:off + MANO_STRIDE]
        out[side] = (bool(h[0]), h[4:7], h[7:52], h[1:4], h[52:62])
    return out


def read_keypoints(path):
    """-> {'l'|'r': (valid, (21,3))} in camera coordinates."""
    v = np.loadtxt(path, dtype=np.float64)
    if v.shape[0] != 2 * KP_STRIDE:
        raise RuntimeError(f'{path}: expected {2*KP_STRIDE} values, got {v.shape[0]}')
    return {side: (bool(v[off]), v[off + 1:off + KP_STRIDE].reshape(21, 3))
            for side, off in (('l', 0), ('r', KP_STRIDE))}


def frame_ids(cam_dir):
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob(os.path.join(cam_dir, 'rgb', '*.png')))


def convert_sequence(cam_dir, out_dir, jpeg_q=92, tol_mm=0.01):
    ids = frame_ids(cam_dir)
    if not ids:
        raise RuntimeError(f'{cam_dir}: no rgb frames')
    T = len(ids)

    fx, fy, cx, cy, W, H = np.loadtxt(os.path.join(cam_dir, 'cam_intrinsics.txt'))
    W, H = int(W), int(H)
    focal = float((fx + fy) / 2.0)          # fx and fy differ by <0.1%
    os.makedirs(os.path.join(out_dir, 'extracted_images'), exist_ok=True)

    anno = {f'{k}_{s}': np.zeros((T, d), dtype=np.float32)
            for k, d in [('rot', 3), ('trans', 3), ('pose', 45), ('betas', 10)]
            for s in ('l', 'r')}
    check = {'l': [], 'r': []}
    n_valid = {'l': 0, 'r': 0}

    for i, fid in enumerate(ids):
        img = cv2.imread(os.path.join(cam_dir, 'rgb', f'{fid}.png'))
        if img is None:
            raise RuntimeError(f'{cam_dir}: unreadable rgb/{fid}.png')
        # Translate so the principal point lands on the image centre, which is
        # what HaWoR's projection assumes. Written as JPEG: the source PNGs are
        # ~1.4MB each and lossless storage buys nothing downstream.
        M = np.float32([[1, 0, W / 2.0 - cx], [0, 1, H / 2.0 - cy]])
        cv2.imwrite(os.path.join(out_dir, 'extracted_images', f'{i:04d}.jpg'),
                    cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR),
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])

        mano = read_mano(os.path.join(cam_dir, 'hand_pose_mano', f'{fid}.txt'))
        kp = read_keypoints(os.path.join(cam_dir, 'hand_pose', f'{fid}.txt'))
        for side in ('l', 'r'):
            valid, rot, pose45, trans, betas = mano[side]
            if not valid:
                continue                       # hand absent -> rot stays 0
            anno[f'rot_{side}'][i] = rot
            anno[f'pose_{side}'][i] = pose45   # already axis-angle
            anno[f'trans_{side}'][i] = trans
            anno[f'betas_{side}'][i] = betas
            n_valid[side] += 1
            k_valid, joints = kp[side]
            if k_valid:
                check[side].append((i, joints))   # already OpenPose order, camera frame

    if not any(n_valid.values()):
        raise RuntimeError(f'{cam_dir}: no annotated frames')

    # --- verify against H2O's own keypoints ---------------------------------
    # Gate on the shipped 3D joints. Note the left hand is checked with
    # fix_shapedirs=False, matching the convention its betas were fitted in and
    # the one the marker below selects at preprocess time -- checking it any
    # other way would test the shapedirs disagreement rather than this code.
    from hawor.utils.process import run_mano, run_mano_left
    errs = {}
    for side, fn in (('l', run_mano_left), ('r', run_mano)):
        if not check[side]:
            continue
        idx = np.array([c[0] for c in check[side]])
        gt = np.stack([c[1] for c in check[side]])
        kw = {'fix_shapedirs': False} if side == 'l' else {}
        out = fn(torch.from_numpy(anno[f'trans_{side}'][idx]).float()[None],
                 torch.from_numpy(anno[f'rot_{side}'][idx]).float()[None],
                 torch.from_numpy(anno[f'pose_{side}'][idx]).float()[None],
                 betas=torch.from_numpy(anno[f'betas_{side}'][idx]).float()[None],
                 use_cuda=False, **kw)
        pred = out['joints'][0].numpy()[:, :21]
        e = np.linalg.norm(pred[:, NON_TIP_JOINTS] - gt[:, NON_TIP_JOINTS], axis=-1) * 1000.0
        if e.mean() > tol_mm:
            raise RuntimeError(
                f'{cam_dir}: {side} check FAILED, mean {e.mean():.5f}mm max '
                f'{e.max():.5f}mm over {len(idx)} frames (tol {tol_mm}mm). '
                f'Refusing to write -- H2O is axis-angle in OpenCV coordinates '
                f'with identity joint order; suspect one of those.')
        errs[side] = float(e.mean())

    # H2O's left-hand betas were fitted against the uncorrected MANO_LEFT
    # shapedirs, as DexYCB's were. See the module docstring.
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
    return T, n_valid['l'], n_valid['r'], errs


def list_sequences(root):
    """(name, cam_dir) for every egocentric sequence under <subject>_ego/."""
    out = []
    for cam in sorted(glob(os.path.join(root, '*_ego', '*', '*', 'cam4'))):
        parts = cam.split(os.sep)
        subject = parts[-4].replace('_ego', '')
        out.append((f'{subject}__{parts[-3]}__{parts[-2]}', cam))
    return out


def official_splits(root):
    """{'train'|'val'|'test': set of sequence names} from label_split/."""
    out = {}
    for split in ('train', 'val', 'test'):
        p = os.path.join(root, 'label_split', f'pose_{split}.txt')
        if not os.path.exists(p):
            continue
        names = set()
        with open(p) as f:
            for line in f:
                # e.g. subject1/h1/0/cam4/rgb/000000.png
                bits = line.strip().split('/')
                if len(bits) >= 4:
                    names.add(f'{bits[0]}__{bits[1]}__{bits[2]}')
        out[split] = names
    return out


def _worker(args):
    name, cam_dir, out_dir = args
    try:
        if os.path.exists(os.path.join(out_dir, 'anno.pth')):
            return name, 'skip', 0, 0, 0, {}
        T, nl, nr, e = convert_sequence(cam_dir, out_dir)
        return name, 'ok', T, nl, nr, e
    except Exception:
        traceback.print_exc()
        return name, 'fail', 0, 0, 0, {}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--h2o_root', required=True,
                   help='Directory holding subject*_ego/ and label_split/')
    p.add_argument('--out_root', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    args = p.parse_args()

    seqs = list_sequences(args.h2o_root)
    if args.limit:
        seqs = seqs[:args.limit]
    if not seqs:
        raise SystemExit(f'No cam4 sequences under {args.h2o_root}')
    os.makedirs(args.out_root, exist_ok=True)
    print(f'{len(seqs)} egocentric sequences')

    jobs = [(n, c, os.path.join(args.out_root, n)) for n, c in seqs]
    done, failed, nl, nr, errs = [], [], 0, 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            name, status, T, a, b, e = fut.result()
            (failed if status == 'fail' else done).append(name)
            nl += a
            nr += b
            errs.extend(e.values())
            if k % 20 == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} ({len(failed)} failed)', flush=True)

    # Prefer the dataset's own splits; anything they do not mention goes to train.
    official = official_splits(args.h2o_root)
    if official:
        assigned = {s: [n for n in sorted(done) if n in names]
                    for s, names in official.items()}
        leftover = [n for n in sorted(done)
                    if not any(n in names for names in official.values())]
        assigned.setdefault('train', []).extend(leftover)
        if leftover:
            print(f'  {len(leftover)} sequences not named in any official split '
                  f'-> train')
    else:
        print('  WARNING: label_split/ not found, writing a single all.json')
        assigned = {'all': sorted(done)}

    for split, names in assigned.items():
        with open(os.path.join(args.out_root, f'{split}.json'), 'w') as f:
            json.dump(sorted(names), f, indent=1)
        print(f'  {split}.json: {len(names)} sequences')

    print(f'converted {len(done)} sequences, {len(failed)} failed')
    print(f'annotated hands: {nl} left / {nr} right')
    if errs:
        print(f'joint check vs H2O keypoints: mean {np.mean(errs):.5f}mm, '
              f'worst {np.max(errs):.5f}mm')


if __name__ == '__main__':
    main()
