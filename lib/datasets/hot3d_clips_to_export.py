"""
Convert HOT3D-Clips (Hugging Face `bop-benchmark/hot3d`) into the export layout
that lib/datasets/hawor_preprocess_train.py consumes.

HOT3D-Clips is the ungated distribution of HOT3D, so this avoids the credentialed
`Hot3DAria_download_urls.json` route entirely. Three format gaps are bridged here:

1. Pose. Clips store 15 MANO PCA coefficients plus a 6-dof wrist transform;
   HaWoR wants 45 axis-angle values, a 3-dof global orientation and a translation.
   The expansion is `thetas @ hands_components[:15] + hands_mean`, which is what
   smplx does internally for use_pca=True / flat_hand_mean=False. Verified against
   hot3d/data_loaders/mano_layer.py to 1e-4 mm on vertices.
2. Camera. The Aria RGB stream is FISHEYE624; HaWoR assumes a single pinhole
   focal length. Frames are undistorted to a linear camera with projectaria_tools.
3. Orientation. Aria RGB is stored rotated. Images are rotated upright here, and
   the camera pose is written so that load_gt_cam's `... @ R_90` recovers exactly
   the upright camera-to-world matrix.

Usage:
    python lib/datasets/hot3d_clips_to_export.py \
        --clips_dir datasets/hot3d_clips/train_aria \
        --out_root datasets/hot3d_clips_export --workers 16
"""
import argparse
import io
import json
import os
import pickle
import sys
import tarfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath('.'))

import cv2
import joblib
import numpy as np
import torch

RGB_STREAM = '214-1'
NUM_PCA = 15
_PCA_CACHE = {}


def mano_pca_basis(side):
    """(components[:15], hands_mean) for one hand, cached per process."""
    if side not in _PCA_CACHE:
        pkl = ('_DATA/data/mano/MANO_RIGHT.pkl' if side == 'right'
               else '_DATA/data_left/mano_left/MANO_LEFT.pkl')
        with open(pkl, 'rb') as f:
            d = pickle.load(f, encoding='latin1')
        _PCA_CACHE[side] = (np.array(d['hands_components'], dtype=np.float64)[:NUM_PCA],
                            np.array(d['hands_mean'], dtype=np.float64))
    return _PCA_CACHE[side]


def thetas_to_axis_angle(thetas, side):
    comps, mean = mano_pca_basis(side)
    return (np.asarray(thetas, dtype=np.float64) @ comps + mean).astype(np.float32)


def quat_trans_to_matrix(q_wxyz, t_xyz):
    """SE3 from a wxyz quaternion and a translation."""
    w, x, y, z = q_wxyz
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3], M[:3, 3] = R, np.asarray(t_xyz, dtype=np.float32)
    return M


def build_src_calibration(calib_json):
    """A projectaria CameraCalibration matching the clip's stored calibration."""
    from projectaria_tools.core import calibration as C
    from projectaria_tools.core.sophus import SE3

    model = getattr(C.CameraModelType, calib_json['projection_model_type'].split('.')[-1])
    T = calib_json['T_device_from_camera']
    T_device_camera = SE3.from_quat_and_translation(
        float(T['quaternion_wxyz'][0]),
        np.array(T['quaternion_wxyz'][1:], dtype=np.float64),
        np.array(T['translation_xyz'], dtype=np.float64),
    )
    return C.CameraCalibration(
        calib_json['label'],
        model,
        np.array(calib_json['projection_params'], dtype=np.float64),
        T_device_camera,
        int(calib_json['image_width']),
        int(calib_json['image_height']),
        None,
        float(calib_json.get('max_solid_angle', 3.14)),
        '',
    )


def convert_clip(tar_path, out_dir, out_size, jpeg_quality=92):
    """Convert one clip tar into an export directory. Returns the frame count."""
    from projectaria_tools.core import calibration as C
    from projectaria_tools.core.image import InterpolationMethod

    os.makedirs(os.path.join(out_dir, 'extracted_images'), exist_ok=True)

    with tarfile.open(tar_path) as tf:
        members = {m.name: m for m in tf.getmembers()}
        shapes = json.load(io.BytesIO(tf.extractfile(members['__hand_shapes.json__']).read()))
        betas = np.array(shapes['mano'], dtype=np.float32)

        frame_ids = sorted({n.split('.')[0] for n in members if n.endswith('.hands.json')})
        T = len(frame_ids)

        anno = {f'{k}_{s}': np.zeros((T, d), dtype=np.float32)
                for k, d in [('rot', 3), ('trans', 3), ('pose', 45), ('betas', 10)]
                for s in ('l', 'r')}
        RT_c2w_native = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))

        dst_calib = None
        focal_out = None
        for i, fid in enumerate(frame_ids):
            cams = json.load(io.BytesIO(tf.extractfile(members[f'{fid}.cameras.json']).read()))
            hands = json.load(io.BytesIO(tf.extractfile(members[f'{fid}.hands.json']).read()))
            cam = cams[RGB_STREAM]

            RT_c2w_native[i] = quat_trans_to_matrix(
                cam['T_world_from_camera']['quaternion_wxyz'],
                cam['T_world_from_camera']['translation_xyz'])

            for side, suffix in (('left', 'l'), ('right', 'r')):
                hand = hands.get(side)
                if not hand or 'mano_pose' not in hand:
                    continue           # leave zeros: preprocessing treats that as invalid
                mp = hand['mano_pose']
                wrist = np.array(mp['wrist_xform'], dtype=np.float32)
                anno[f'rot_{suffix}'][i] = wrist[:3]
                anno[f'trans_{suffix}'][i] = wrist[3:]
                anno[f'pose_{suffix}'][i] = thetas_to_axis_angle(mp['thetas'], side)
                anno[f'betas_{suffix}'][i] = betas

            if dst_calib is None:
                src_calib = build_src_calibration(cam['calibration'])
                w = int(cam['calibration']['image_width'])
                scale = out_size / w
                focal_out = float(cam['calibration']['projection_params'][0]) * scale
                dst_calib = C.get_linear_camera_calibration(
                    out_size, out_size, focal_out, 'camera-rgb-linear')

            buf = np.frombuffer(tf.extractfile(members[f'{fid}.image_{RGB_STREAM}.jpg']).read(),
                                dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img.shape[0] != out_size:
                img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
            und = C.distort_by_calibration(img, dst_calib, _scaled(src_calib, out_size),
                                           InterpolationMethod.BILINEAR)
            # Aria RGB is stored rotated; load_gt_cam applies R_90 to the pose, so
            # the image must be rotated to match.
            und = cv2.rotate(und, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(out_dir, 'extracted_images', f'{i:04d}.jpg'), und,
                        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])

    # load_gt_cam computes head_pose @ ego_extrinsics @ R_90, and we want that to
    # equal the upright camera-to-world, so hand it the native pose directly.
    with open(os.path.join(out_dir, 'head_pose.pkl'), 'wb') as f:
        pickle.dump(RT_c2w_native, f)
    with open(os.path.join(out_dir, 'ego_extrinsics.pkl'), 'wb') as f:
        pickle.dump(np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)), f)
    with open(os.path.join(out_dir, 'focal.txt'), 'w') as f:
        f.write(str(focal_out))
    joblib.dump({k: torch.from_numpy(v) for k, v in anno.items()},
                os.path.join(out_dir, 'anno.pth'))
    return T


def _scaled(src_calib, out_size):
    """Rescale the source calibration to match a resized image."""
    w = int(src_calib.get_image_size()[0])
    if w == int(out_size):
        return src_calib
    s = out_size / float(w)
    return src_calib.rescale(np.array([out_size, out_size], dtype=np.int32), s)


def _worker(args):
    tar_path, out_dir, out_size = args
    name = os.path.splitext(os.path.basename(tar_path))[0]
    try:
        if os.path.exists(os.path.join(out_dir, 'anno.pth')):
            return name, 'skip', 0
        n = convert_clip(tar_path, out_dir, out_size)
        return name, 'ok', n
    except Exception:
        traceback.print_exc()
        return name, 'fail', 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--clips_dir', required=True, help='Directory of clip-*.tar files')
    p.add_argument('--out_root', required=True)
    p.add_argument('--out_size', type=int, default=704,
                   help='Undistorted square image size. 704 halves the native 1408 '
                        'while keeping hand crops near their native resolution.')
    p.add_argument('--limit', type=int, default=None, help='Convert only the first N clips')
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--split', type=str, default='train', help='Manifest name (<split>.json)')
    args = p.parse_args()

    tars = sorted(f for f in os.listdir(args.clips_dir) if f.endswith('.tar'))
    if args.limit:
        tars = tars[:args.limit]
    os.makedirs(args.out_root, exist_ok=True)
    jobs = [(os.path.join(args.clips_dir, t),
             os.path.join(args.out_root, os.path.splitext(t)[0]),
             args.out_size) for t in tars]

    done, failed, frames = [], [], 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_worker, j): j for j in jobs}
        for k, fut in enumerate(as_completed(futs), 1):
            name, status, n = fut.result()
            if status == 'fail':
                failed.append(name)
            else:
                done.append(name)
                frames += n
            if k % 25 == 0 or k == len(jobs):
                print(f'  {k}/{len(jobs)} clips  ({len(failed)} failed)', flush=True)

    manifest = os.path.join(args.out_root, f'{args.split}.json')
    with open(manifest, 'w') as f:
        json.dump(sorted(done), f, indent=1)
    print(f'converted {len(done)} clips ({frames} new frames), {len(failed)} failed')
    print(f'wrote {manifest}')
    if failed:
        print('failed: ' + ', '.join(failed[:10]))


if __name__ == '__main__':
    main()
