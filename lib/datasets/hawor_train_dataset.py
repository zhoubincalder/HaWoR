"""
Chunked training dataset for the camera-space hand motion estimator.

Each sample is a temporally contiguous window of `seq_len` frames of a single
hand, which is what the space-time and motion modules in lib/models/hawor.py
expect. The emitted keys line up exactly with HAWOR.forward_step / compute_loss.

Two conventions matter here:

1. The network is right-hand-only. Left hands are mirrored into right-hand space
   (image, principal point, box center, 2D joints, x of the 3D joints, and the
   y/z components of every axis-angle rotation) and are then treated as ordinary
   right hands. We deliberately do NOT set the `do_flip` key: that flag is an
   inference-time convenience that maps predictions back to the original image,
   and using it during training would put the 3D joints and the projection in
   two different frames.

2. Augmentation is sampled once per window, not per frame. Per-frame sampling
   would inject synthetic jitter that the temporal modules would then learn to
   undo.
"""
import json
import os
import sys
from glob import glob

sys.path.append(os.path.abspath('.'))

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, ToTensor

from lib.core import constants
from lib.utils.imutils import boxes_2_cs, crop

NUM_JOINTS = 21


class HaworChunkDataset(Dataset):
    """
    Args:
        video_root (str): root of the exported sequences.
        set_file (str): JSON list of sequence names, relative to video_root.
        cfg (CfgNode): full config; DATASETS.CONFIG and MODEL.IMAGE_SIZE are read.
        seq_len (int): frames per sample. Must stay 16 -- forward_step hardcodes it.
        stride (int): step between consecutive window start frames.
        train (bool): enables augmentation.
    """

    def __init__(self, video_root, set_file, cfg, seq_len=16, stride=8, train=True,
                 anno_name='train_anno.npz'):
        super().__init__()
        self.video_root = video_root
        self.cfg = cfg
        self.seq_len = seq_len
        self.train = train
        self.anno_name = anno_name
        self.crop_size = cfg.MODEL.IMAGE_SIZE

        aug = cfg.DATASETS.CONFIG
        self.scale_factor = aug.SCALE_FACTOR
        self.trans_factor = aug.TRANS_FACTOR
        self.color_scale = aug.COLOR_SCALE
        self.trans_aug_rate = aug.TRANS_AUG_RATE
        # The crop is dilated the same way as at inference time
        # (TrackDatasetEval is constructed with dilate=1.2 in HAWOR.inference).
        self.box_dilate = cfg.DATASETS.CONFIG.get('BOX_DILATE', 1.2)

        # In-plane rotation augmentation is not supported: the CLIFF bbox feature
        # and the full-frame projection in HAWOR.project both assume an
        # axis-aligned crop, so rotating it would desynchronize the 2D loss from
        # the predicted camera translation.
        if aug.get('ROT_AUG_RATE', 0) > 0 and aug.get('ROT_FACTOR', 0) > 0:
            raise ValueError(
                'ROT_AUG_RATE must be 0 for HaWoR: in-plane crop rotation is '
                'inconsistent with the CLIFF bbox feature and full-frame projection.')
        # Horizontal flip is a no-op for a handed model: flipping a right hand
        # produces a left hand, which this dataset would mirror straight back.
        if aug.get('DO_FLIP', False):
            raise ValueError('DO_FLIP must be False: left hands are already mirrored '
                             'into right-hand space by this dataset.')

        self.normalize_img = Compose([
            ToTensor(),
            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD),
        ])

        with open(os.path.join(video_root, set_file), 'r') as f:
            self.videos = json.load(f)

        self._anno_cache = {}
        self._imgfile_cache = {}
        self.index = self._build_index(stride)
        print(f'[HaworChunkDataset] {len(self.videos)} sequences -> '
              f'{len(self.index)} windows of {seq_len} frames (train={train})')

    # ------------------------------------------------------------------ index

    def _anno_path(self, video):
        return os.path.join(self.video_root, video, self.anno_name)

    def _build_index(self, stride):
        index = []
        for v_idx, video in enumerate(self.videos):
            path = self._anno_path(video)
            if not os.path.exists(path):
                print(f'WARNING: missing {path}, skipping. Run hawor_preprocess_train.py first.')
                continue
            with np.load(path) as data:
                valid = data['valid']
            for hand in (0, 1):
                for start in self._windows(valid[hand], stride):
                    index.append((v_idx, hand, start))
        return index

    def _windows(self, valid, stride):
        """Start frames of fully-valid contiguous windows."""
        starts = []
        T = len(valid)
        run_start = None
        for t in range(T + 1):
            is_valid = t < T and valid[t]
            if is_valid and run_start is None:
                run_start = t
            elif not is_valid and run_start is not None:
                run_len = t - run_start
                if run_len >= self.seq_len:
                    last = run_start + run_len - self.seq_len
                    starts.extend(range(run_start, last + 1, stride))
                    if (last - run_start) % stride != 0:
                        starts.append(last)
                run_start = None
        return starts

    # ------------------------------------------------------------------- data

    def _get_anno(self, video):
        if video not in self._anno_cache:
            with np.load(self._anno_path(video)) as data:
                self._anno_cache[video] = {k: data[k] for k in data.files}
        return self._anno_cache[video]

    def _get_imgfiles(self, video):
        if video not in self._imgfile_cache:
            folder = os.path.join(self.video_root, video, 'extracted_images')
            files = sorted(glob(os.path.join(folder, '*.jpg')))
            if len(files) == 0:
                files = sorted(glob(os.path.join(folder, '*.png')))
            self._imgfile_cache[video] = files
        return self._imgfile_cache[video]

    def _sample_augmentation(self):
        if not self.train:
            return 1.0, np.zeros(2, dtype=np.float32), np.ones(3, dtype=np.float32)
        scale_aug = 1.0 + np.random.uniform(-self.scale_factor, self.scale_factor)
        if np.random.rand() < self.trans_aug_rate:
            trans_aug = np.random.uniform(-self.trans_factor, self.trans_factor, size=2).astype(np.float32)
        else:
            trans_aug = np.zeros(2, dtype=np.float32)
        color_aug = np.random.uniform(1 - self.color_scale, 1 + self.color_scale, size=3).astype(np.float32)
        return scale_aug, trans_aug, color_aug

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        v_idx, hand, start = self.index[idx]
        video = self.videos[v_idx]
        anno = self._get_anno(video)
        imgfiles = self._get_imgfiles(video)
        frames = np.arange(start, start + self.seq_len)

        W, H = int(anno['img_size'][0]), int(anno['img_size'][1])
        img_focal = float(anno['img_focal'])
        img_center = anno['img_center'].astype(np.float32).copy()

        boxes = anno['boxes'][hand, frames].astype(np.float32).copy()
        j2d = anno['j2d'][hand, frames].astype(np.float32).copy()
        j2d_conf = anno['j2d_conf'][hand, frames].astype(np.float32).copy()
        j3d_wo_trans = anno['j3d_wo_trans'][hand, frames].astype(np.float32).copy()
        cam_pose = anno['cam_pose'][hand, frames].astype(np.float32).copy()
        betas = anno['betas'][hand, frames].astype(np.float32).copy()

        centers, scales = boxes_2_cs(boxes)
        centers = centers.astype(np.float32)
        scales = scales.astype(np.float32) * self.box_dilate

        do_mirror = (hand == 0)  # left hand -> mirror into right-hand space

        # --- augmentation, sampled once for the whole window ---
        scale_aug, trans_aug, color_aug = self._sample_augmentation()
        scales = scales * scale_aug
        centers = centers + trans_aug[None, :] * (scales[:, None] * 200.0)

        imgs = []
        for i, t in enumerate(frames):
            img = cv2.imread(imgfiles[t])[:, :, ::-1]
            center = centers[i]
            if do_mirror:
                img = img[:, ::-1, :]
                center = center.copy()
                center[0] = W - 1 - center[0]
                centers[i] = center
            img_crop = crop(img, center, scales[i], [self.crop_size, self.crop_size], rot=0)
            if self.train:
                img_crop = img_crop * color_aug[None, None, :]
            img_crop = np.clip(img_crop, 0, 255).astype('uint8')
            imgs.append(self.normalize_img(img_crop))

        if do_mirror:
            # u -> W-1-u in the image, x -> -x in 3D, and axis-angle vectors
            # transform as pseudovectors: (rx, ry, rz) -> (rx, -ry, -rz).
            img_center[0] = W - 1 - img_center[0]
            j2d[..., 0] = W - 1 - j2d[..., 0]
            j3d_wo_trans[..., 0] *= -1
            cam_pose = cam_pose.reshape(self.seq_len, 16, 3)
            cam_pose[..., 1] *= -1
            cam_pose[..., 2] *= -1
            cam_pose = cam_pose.reshape(self.seq_len, 48)

        # Full-image pixels -> crop pixels -> normalized [-0.5, 0.5], matching
        # the normalization applied to the prediction in HAWOR.forward_step.
        b = scales * 200.0
        j2d_crop = j2d - (centers - b[:, None] / 2)[:, None, :]
        j2d_crop = j2d_crop * (self.crop_size / b)[:, None, None]
        j2d_norm = j2d_crop / self.crop_size - 0.5

        item = {
            'img': torch.stack(imgs).float(),
            'center': torch.from_numpy(centers).float(),
            'scale': torch.from_numpy(scales).float(),
            'img_focal': torch.full((self.seq_len,), img_focal).float(),
            'img_center': torch.from_numpy(img_center).float().unsqueeze(0).repeat(self.seq_len, 1),
            'gt_cam_j2d': torch.from_numpy(j2d_norm).float(),
            'gt_cam_j2d_conf': torch.from_numpy(j2d_conf).float(),
            'gt_j3d_wo_trans': torch.from_numpy(j3d_wo_trans).float(),
            'gt_cam_full_pose': torch.from_numpy(cam_pose).float(),
            'gt_cam_betas': torch.from_numpy(betas).float(),
        }
        return item
