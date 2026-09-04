"""
Offline ground-truth extraction for training the camera-space hand motion estimator.

For every exported sequence this writes a single `train_anno.npz` holding, per hand
and per frame, everything the training dataset needs:

    valid        (2, T)          frame usable for training
    boxes        (2, T, 4)       xyxy box derived from the projected GT joints
    j2d          (2, T, 21, 2)   GT joints in full-image pixels
    j2d_conf     (2, T, 21)      1.0 when the joint is in front of the camera and inside the image
    j3d_wo_trans (2, T, 21, 3)   GT joints from MANO with camera-space global orient and zero translation
    cam_pose     (2, T, 48)      camera-space axis-angle: root orient + 15 finger joints
    betas        (2, T, 10)

Hand index 0 is the left hand and 1 is the right hand, matching the rest of the repo.
Everything stays in true camera space here -- the left/right mirroring that the
right-hand-only network needs is applied on the fly by the training dataset.

Usage:
    python lib/datasets/hawor_preprocess_train.py \
        --video_root datasets/hot3d_trainset_export --set_file train.json
"""
import argparse
import json
import os
import sys
from glob import glob

sys.path.append(os.path.abspath('.'))

import cv2
import joblib
import numpy as np
import torch
from tqdm import tqdm

from hawor.utils.geometry import aa_to_rotmat
from hawor.utils.process import run_mano, run_mano_left
from hawor.utils.rotation import rotation_matrix_to_angle_axis
from lib.eval_utils.custom_utils import load_gt_cam

NUM_JOINTS = 21
HANDS = ['left', 'right']

# A sequence directory containing this file has left-hand betas fitted against
# the UNCORRECTED MANO_LEFT shapedirs (manopth's convention -- DexYCB and
# anything else built on manopth). Written by the converter, read here.
MANO_LEFT_UNFIXED_MARKER = 'mano_left_unfixed'


def load_image_files(video_dir):
    img_folder = os.path.join(video_dir, 'extracted_images')
    imgfiles = sorted(glob(os.path.join(img_folder, '*.jpg')))
    if len(imgfiles) == 0:
        imgfiles = sorted(glob(os.path.join(img_folder, '*.png')))
    return imgfiles


def run_mano_batched(hand, trans, root_orient, hand_pose, betas, use_cuda=True,
                     chunk=1024, fix_shapedirs=True):
    """Run MANO over a long sequence in chunks so we do not blow up GPU memory.

    fix_shapedirs applies only to the left hand: MANO_LEFT.pkl ships mirrored
    shapedirs (smplx issue #48) and run_mano_left corrects them by default. A
    dataset whose left-hand betas were fitted against the UNCORRECTED model --
    anything built on manopth, DexYCB included -- must be evaluated with the
    correction off, or its shape is reinterpreted and the joints move ~15mm.
    See MANO_LEFT_UNFIXED_MARKER.
    """
    fn = run_mano if hand == 'right' else run_mano_left
    kw = {} if hand == 'right' else {'fix_shapedirs': fix_shapedirs}
    joints = []
    T = trans.shape[1]
    for s in range(0, T, chunk):
        e = min(T, s + chunk)
        out = fn(trans[:, s:e], root_orient[:, s:e], hand_pose[:, s:e],
                 betas=betas[:, s:e], use_cuda=use_cuda, **kw)
        joints.append(out['joints'][0].detach().cpu())
    return torch.cat(joints, dim=0)  # (T, 21, 3)


def boxes_from_j2d(j2d, conf, W, H, pad=0.2):
    """GT-driven boxes, same 20% padding convention as lib/pipeline/tools.py."""
    T = j2d.shape[0]
    boxes = np.zeros((T, 4), dtype=np.float32)
    n_vis = np.zeros((T,), dtype=np.int64)
    for t in range(T):
        vis = conf[t] > 0
        n_vis[t] = int(vis.sum())
        if n_vis[t] < 2:
            continue
        x, y = j2d[t, vis, 0], j2d[t, vis, 1]
        det_w, det_h = x.max() - x.min(), y.max() - y.min()
        boxes[t] = [
            max(0, x.min() - pad * det_w),
            max(0, y.min() - pad * det_h),
            min(W, x.max() + pad * det_w),
            min(H, y.max() + pad * det_h),
        ]
    return boxes, n_vis


def process_video(video_root, video, min_vis_joints=12, use_cuda=True, overwrite=False):
    video_dir = os.path.join(video_root, video)
    out_path = os.path.join(video_dir, 'train_anno.npz')
    if os.path.exists(out_path) and not overwrite:
        print(f'skip {video} (already processed)')
        return out_path

    imgfiles = load_image_files(video_dir)
    if len(imgfiles) == 0:
        raise FileNotFoundError(f'No extracted images under {video_dir}/extracted_images')
    img = cv2.imread(imgfiles[0])
    H, W = img.shape[:2]

    with open(os.path.join(video_dir, 'focal.txt'), 'r') as f:
        img_focal = float(f.read())
    # [cx, cy]. Note scripts/scripts_eval/eval_hawor_hot3d.py builds this as
    # [h/2, w/2]; the model reads index 0 as x, so [w/2, h/2] is the correct order.
    img_center = np.array([W / 2.0, H / 2.0], dtype=np.float32)
    K = np.array([[img_focal, 0, img_center[0]],
                  [0, img_focal, img_center[1]],
                  [0, 0, 1]], dtype=np.float32)

    fix_shapedirs = not os.path.exists(
        os.path.join(video_dir, MANO_LEFT_UNFIXED_MARKER))

    anno = joblib.load(os.path.join(video_dir, 'anno.pth'))
    world_rot = torch.stack([anno['rot_l'], anno['rot_r']]).float()          # (2, T, 3)
    world_trans = torch.stack([anno['trans_l'], anno['trans_r']]).float()    # (2, T, 3)
    world_pose = torch.stack([anno['pose_l'], anno['pose_r']]).float()       # (2, T, 45)
    world_betas = torch.stack([anno['betas_l'], anno['betas_r']]).float()    # (2, T, 10)
    mano_valid = torch.any(world_rot != 0, dim=-1).numpy()                   # (2, T)

    T = world_rot.shape[1]
    T = min(T, len(imgfiles))
    world_rot, world_trans = world_rot[:, :T], world_trans[:, :T]
    world_pose, world_betas = world_pose[:, :T], world_betas[:, :T]
    mano_valid = mano_valid[:, :T]

    R_w2c, t_w2c, _, _ = load_gt_cam(video_root, video)
    R_w2c, t_w2c = R_w2c[:T].float(), t_w2c[:T].float()

    valid = np.zeros((2, T), dtype=bool)
    boxes = np.zeros((2, T, 4), dtype=np.float32)
    j2d_all = np.zeros((2, T, NUM_JOINTS, 2), dtype=np.float32)
    conf_all = np.zeros((2, T, NUM_JOINTS), dtype=np.float32)
    j3d_wo_trans_all = np.zeros((2, T, NUM_JOINTS, 3), dtype=np.float32)
    cam_pose_all = np.zeros((2, T, 48), dtype=np.float32)
    betas_all = world_betas.numpy().astype(np.float32)

    for h_idx, hand in enumerate(HANDS):
        rot = world_rot[h_idx:h_idx + 1]
        trans = world_trans[h_idx:h_idx + 1]
        pose = world_pose[h_idx:h_idx + 1]
        betas = world_betas[h_idx:h_idx + 1]

        # World-space joints -> camera space, used for the 2D reprojection target.
        world_joints = run_mano_batched(hand, trans, rot, pose, betas,
                                        use_cuda=use_cuda, fix_shapedirs=fix_shapedirs)
        cam_j3d = torch.einsum('tij,tnj->tni', R_w2c, world_joints) + t_w2c[:, None, :]

        # Camera-space root orientation; finger joint rotations are relative and unchanged.
        R_root_world = aa_to_rotmat(rot.reshape(-1, 3))
        R_root_cam = torch.einsum('tij,tjk->tik', R_w2c, R_root_world)
        root_cam_aa = rotation_matrix_to_angle_axis(R_root_cam)  # (T, 3)

        # Joints with camera-space orientation but no translation: exactly what the
        # network's MANO head outputs, and what the 3D loss compares against.
        j3d_wo_trans = run_mano_batched(
            hand,
            torch.zeros(1, T, 3),
            root_cam_aa[None],
            pose,
            betas,
            use_cuda=use_cuda,
            fix_shapedirs=fix_shapedirs,
        )

        # Project into the full image.
        z = cam_j3d[..., 2].clamp(min=1e-4)
        u = float(K[0, 0]) * cam_j3d[..., 0] / z + float(K[0, 2])
        v = float(K[1, 1]) * cam_j3d[..., 1] / z + float(K[1, 2])
        j2d = torch.stack([u, v], dim=-1).numpy()
        in_front = (cam_j3d[..., 2] > 0).numpy()
        in_image = (j2d[..., 0] >= 0) & (j2d[..., 0] < W) & (j2d[..., 1] >= 0) & (j2d[..., 1] < H)
        conf = (in_front & in_image).astype(np.float32)

        hand_boxes, n_vis = boxes_from_j2d(j2d, conf, W, H)

        j2d_all[h_idx] = j2d
        conf_all[h_idx] = conf
        j3d_wo_trans_all[h_idx] = j3d_wo_trans.numpy()
        cam_pose_all[h_idx] = np.concatenate(
            [root_cam_aa.numpy(), pose[0].numpy()], axis=-1).astype(np.float32)
        boxes[h_idx] = hand_boxes
        valid[h_idx] = mano_valid[h_idx] & (n_vis >= min_vis_joints)

    np.savez_compressed(
        out_path,
        valid=valid,
        boxes=boxes,
        j2d=j2d_all,
        j2d_conf=conf_all,
        j3d_wo_trans=j3d_wo_trans_all,
        cam_pose=cam_pose_all,
        betas=betas_all,
        img_focal=np.float32(img_focal),
        img_center=img_center,
        img_size=np.array([W, H], dtype=np.int64),
        n_frames=np.int64(T),
    )
    print(f'{video}: {valid[0].sum()} left / {valid[1].sum()} right valid frames of {T}')
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_root', type=str, required=True,
                        help='Root of the exported sequences, e.g. datasets/hot3d_trainset_export')
    parser.add_argument('--set_file', type=str, default='train.json',
                        help='JSON list of sequence names, relative to --video_root')
    parser.add_argument('--min_vis_joints', type=int, default=12,
                        help='Minimum number of in-image joints for a frame to be trainable')
    parser.add_argument('--cpu', action='store_true', help='Run MANO on CPU')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    with open(os.path.join(args.video_root, args.set_file), 'r') as f:
        videos = json.load(f)

    for video in tqdm(videos):
        process_video(args.video_root, video,
                      min_vis_joints=args.min_vis_joints,
                      use_cuda=not args.cpu,
                      overwrite=args.overwrite)


if __name__ == '__main__':
    main()
