"""
Convert ARCTIC's egocentric split into the export layout that
lib/datasets/hawor_preprocess_train.py consumes.

ARCTIC is easier to bridge than HOT3D-Clips: `raw_seqs/*.mano.npy` already stores
full 45-dim axis-angle hand pose, a 3-dim global orientation and a translation in
world coordinates, so no PCA expansion is needed. Three things still need care:

1. Camera. `raw_seqs/*.egocam.dist.npy` gives per-frame world-to-camera rotation
   and translation plus an 8-parameter OpenCV rational distortion model
   (k1,k2,p1,p2,k3,k4,k5,k6 -- verified against common/transforms.py:102-107).
   HaWoR assumes an undistorted pinhole camera with a single focal length and the
   principal point at the image centre, so frames are undistorted into exactly
   that camera. ARCTIC's principal point is off-centre (cx=1328.7 for a 2800-wide
   image), which is why re-centring matters rather than just undistorting in place.

2. Image scale. `cropped_images` are only object-centre cropped for the
   *allocentric* views. The egocentric view (view 0) is merely resized by
   EGO_IMAGE_SCALE = 0.3 -- see scripts_data/crop_images.py:52-57 -- so intrinsics
   scale by 0.3 with no crop offset to track.

3. Frame indexing. Image files are 1-indexed by the per-subject `ioi_offset` from
   meta/misc.json: image `%05d.jpg` holds annotation index `idx - ioi_offset`.

Usage:
    python lib/datasets/arctic_to_export.py \
        --arctic_root ~/ws/arctic --out_root datasets/arctic_export --workers 8
"""
import argparse
import io
import json
import os
import pickle
import sys
import traceback
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath('.'))

import cv2
import joblib
import numpy as np
import torch

EGO_VIEW = '0'
_MEAN_CACHE = {}
EGO_IMAGE_SCALE = 0.3

# load_gt_cam() computes head_pose @ ego_extrinsics @ R_90. ARCTIC frames are
# already upright, so the pose is pre-multiplied by R_90's inverse to cancel it.
R_90 = np.array([[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float64)


def mano_hands_mean(side):
    """MANO's mean finger pose (45,), cached per process.

    ARCTIC builds its MANO layer with flat_hand_mean=False and calls it with
    pose2rot=True, so smplx adds this mean to the stored `pose` internally. HaWoR's
    run_mano passes rotation matrices (pose2rot=False), which skips that addition,
    so the mean has to be folded in here. Omitting it displaces vertices by up to
    ~75mm (right) / ~98mm (left).
    """
    if side not in _MEAN_CACHE:
        pkl = ('_DATA/data/mano/MANO_RIGHT.pkl' if side == 'right'
               else '_DATA/data_left/mano_left/MANO_LEFT.pkl')
        with open(pkl, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        _MEAN_CACHE[side] = np.array(d['hands_mean'], dtype=np.float32).reshape(45)
    return _MEAN_CACHE[side]

def list_sequences(arctic_root):
    """Sequences that have both raw annotations and a cropped-image archive."""
    raw = os.path.join(arctic_root, 'unpacked', 'raw_seqs')
    zips = os.path.join(arctic_root, 'downloads', 'data', 'cropped_images_zips')
    out = []
    for sid in sorted(os.listdir(raw)) if os.path.isdir(raw) else []:
        for f in sorted(os.listdir(os.path.join(raw, sid))):
            if not f.endswith('.mano.npy'):
                continue
            seq = f[:-len('.mano.npy')]
            z = os.path.join(zips, sid, f'{seq}.zip')
            if os.path.exists(z) and os.path.getsize(z) > 1024:
                out.append((sid, seq, z))
    return out


def convert_sequence(arctic_root, sid, seq, zip_path, out_dir, ioi_offset):
    raw = os.path.join(arctic_root, 'unpacked', 'raw_seqs', sid)
    mano = np.load(os.path.join(raw, f'{seq}.mano.npy'), allow_pickle=True).item()
    ego = np.load(os.path.join(raw, f'{seq}.egocam.dist.npy'), allow_pickle=True).item()

    R_w2c = np.asarray(ego['R_k_cam_np'], dtype=np.float64)          # (T,3,3)
    t_w2c = np.asarray(ego['T_k_cam_np'], dtype=np.float64).reshape(-1, 3)
    K_full = np.asarray(ego['intrinsics'], dtype=np.float64)
    dist = np.asarray(ego['dist8'], dtype=np.float64).reshape(-1)
    T_anno = R_w2c.shape[0]

    os.makedirs(os.path.join(out_dir, 'extracted_images'), exist_ok=True)

    with zipfile.ZipFile(zip_path) as zf:
        names = {n for n in zf.namelist() if n.endswith('.jpg') and f'/{EGO_VIEW}/' in f'/{n}'}
        by_idx = {}
        for n in names:
            parts = n.split('/')
            if parts[-2] != EGO_VIEW:
                continue
            try:
                by_idx[int(os.path.splitext(parts[-1])[0])] = n
            except ValueError:
                pass
        if not by_idx:
            raise RuntimeError(f'{zip_path}: no view-{EGO_VIEW} frames')

        # Annotation index t <-> image file (t + ioi_offset)
        usable = sorted(t for t in range(T_anno) if (t + ioi_offset) in by_idx)
        if not usable:
            raise RuntimeError(f'{zip_path}: no overlap between images and annotations')

        first = cv2.imdecode(np.frombuffer(zf.read(by_idx[usable[0] + ioi_offset]), np.uint8),
                             cv2.IMREAD_COLOR)
        H, W = first.shape[:2]

        # Intrinsics of the on-disk (already 0.3-scaled) image.
        K = K_full.copy()
        K[:2, :] *= EGO_IMAGE_SCALE
        focal = float((K[0, 0] + K[1, 1]) / 2.0)
        # Target: single focal length, principal point at the image centre.
        K_new = np.array([[focal, 0, W / 2.0], [0, focal, H / 2.0], [0, 0, 1]], dtype=np.float64)
        map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K_new, (W, H), cv2.CV_16SC2)

        n_out = len(usable)
        anno = {f'{k}_{s}': np.zeros((n_out, d), dtype=np.float32)
                for k, d in [('rot', 3), ('trans', 3), ('pose', 45), ('betas', 10)]
                for s in ('l', 'r')}
        RT_c2w = np.tile(np.eye(4, dtype=np.float64), (n_out, 1, 1))

        for i, t in enumerate(usable):
            img = cv2.imdecode(np.frombuffer(zf.read(by_idx[t + ioi_offset]), np.uint8),
                               cv2.IMREAD_COLOR)
            und = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(out_dir, 'extracted_images', f'{i:04d}.jpg'), und,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            R, tt = R_w2c[t], t_w2c[t]
            RT_c2w[i, :3, :3] = R.T
            RT_c2w[i, :3, 3] = -R.T @ tt

            for side, suffix in (('left', 'l'), ('right', 'r')):
                m = mano[side]
                anno[f'rot_{suffix}'][i] = m['rot'][t]
                # ARCTIC's pose is an offset from MANO's hands_mean; see mano_hands_mean().
                anno[f'pose_{suffix}'][i] = m['pose'][t] + mano_hands_mean(side)
                anno[f'trans_{suffix}'][i] = m['trans'][t]
                anno[f'betas_{suffix}'][i] = np.asarray(m['shape'], dtype=np.float32)

    head_pose = np.einsum('tij,jk->tik', RT_c2w, np.linalg.inv(R_90)).astype(np.float32)
    with open(os.path.join(out_dir, 'head_pose.pkl'), 'wb') as f:
        pickle.dump(head_pose, f)
    with open(os.path.join(out_dir, 'ego_extrinsics.pkl'), 'wb') as f:
        pickle.dump(np.tile(np.eye(4, dtype=np.float32), (n_out, 1, 1)), f)
    with open(os.path.join(out_dir, 'focal.txt'), 'w') as f:
        f.write(str(focal))
    joblib.dump({k: torch.from_numpy(v) for k, v in anno.items()},
                os.path.join(out_dir, 'anno.pth'))
    return n_out


def _worker(args):
    arctic_root, sid, seq, zip_path, out_dir, ioi_offset = args
    name = f'{sid}__{seq}'
    try:
        if os.path.exists(os.path.join(out_dir, 'anno.pth')):
            return name, 'skip', 0
        n = convert_sequence(arctic_root, sid, seq, zip_path, out_dir, ioi_offset)
        return name, 'ok', n
    except Exception:
        traceback.print_exc()
        return name, 'fail', 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arctic_root', default=os.path.expanduser('~/ws/arctic'))
    p.add_argument('--out_root', required=True)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--split', default='train', help='Manifest name (<split>.json)')
    args = p.parse_args()

    with open(os.path.join(args.arctic_root, 'unpacked', 'meta', 'misc.json')) as f:
        misc = json.load(f)

    seqs = list_sequences(args.arctic_root)
    if args.limit:
        seqs = seqs[:args.limit]
    if not seqs:
        raise SystemExit('No sequences with both raw_seqs and a cropped-images zip. '
                         'Unzip raw_seqs.zip/meta.zip into <arctic_root>/unpacked first.')
    os.makedirs(args.out_root, exist_ok=True)

    jobs = [(args.arctic_root, sid, seq, z,
             os.path.join(args.out_root, f'{sid}__{seq}'), misc[sid]['ioi_offset'])
            for sid, seq, z in seqs]

    done, failed, frames = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_worker, j) for j in jobs]
        for k, fut in enumerate(as_completed(futs), 1):
            name, status, n = fut.result()
            (failed if status == 'fail' else done).append(name)
            frames += n
            if k % 10 == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} sequences ({len(failed)} failed)', flush=True)

    manifest = os.path.join(args.out_root, f'{args.split}.json')
    with open(manifest, 'w') as f:
        json.dump(sorted(done), f, indent=1)
    print(f'converted {len(done)} sequences ({frames} frames), {len(failed)} failed')
    print(f'wrote {manifest}')


if __name__ == '__main__':
    main()
