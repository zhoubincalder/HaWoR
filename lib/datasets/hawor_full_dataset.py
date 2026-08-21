"""
Full-frame training dataset: both hands from one image, no crops, no detector.

Differences from the per-hand crop dataset (hawor_train_dataset.py):

- One sample is a 16-frame window of *whole frames* carrying BOTH hands, so a
  single backbone pass yields both. There is no per-hand mirroring, because both
  hands appear in the same image and cannot be mirrored independently.
- Frames are letterboxed to the model input, which keeps the camera a simple
  uniform scale plus offset: f' = f*s, c' = c*s + pad. Anisotropic resizing would
  make the two focal lengths differ, which HaWoR's single-focal projection cannot
  express.
- Ground truth keeps a leading hand axis (slot 0 left, slot 1 right) plus a
  per-hand validity flag, since on a full frame a hand is often absent -- with
  crops that case never reached the model.
- Only colour augmentation. Scale/translation jitter existed to perturb the crop
  box; there is no crop here, and warping the frame would change the intrinsics.
"""
import json
import os
import sys
from glob import glob

sys.path.append(os.path.abspath('.'))

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor

from lib.core import constants

NUM_JOINTS = 21


class HaworFullDataset(torch.utils.data.Dataset):
    def __init__(self, video_root, set_file, cfg, seq_len=16, stride=8, train=True,
                 anno_name='train_anno.npz'):
        super().__init__()
        self.video_root = video_root
        self.seq_len = seq_len
        self.train = train
        self.anno_name = anno_name
        self.in_h = cfg.MODEL.get('INPUT_H', 384)
        self.in_w = cfg.MODEL.get('INPUT_W', 512)
        self.color_scale = cfg.DATASETS.CONFIG.get('COLOR_SCALE', 0.2)

        self.normalize_img = Compose([
            ToTensor(),
            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD),
        ])
        with open(os.path.join(video_root, set_file)) as f:
            self.videos = json.load(f)
        self._anno, self._imgs = {}, {}
        self.index = self._build_index(stride)
        print(f'[HaworFullDataset] {len(self.videos)} sequences -> {len(self.index)} '
              f'windows of {seq_len} frames at {self.in_h}x{self.in_w} (train={train})')

    # ---------------------------------------------------------------- indexing
    def _path(self, v):
        return os.path.join(self.video_root, v, self.anno_name)

    def _build_index(self, stride):
        idx = []
        for vi, v in enumerate(self.videos):
            p = self._path(v)
            if not os.path.exists(p):
                continue
            with np.load(p) as d:
                valid = d['valid']
            # A window is usable if at least one hand is valid in every frame;
            # the per-hand flags handle the rest.
            any_hand = valid.any(axis=0)
            T = len(any_hand)
            run = None
            for t in range(T + 1):
                ok = t < T and any_hand[t]
                if ok and run is None:
                    run = t
                elif not ok and run is not None:
                    if t - run >= self.seq_len:
                        last = run + (t - run) - self.seq_len
                        idx += [(vi, s) for s in range(run, last + 1, stride)]
                        if (last - run) % stride:
                            idx.append((vi, last))
                    run = None
        return idx

    def _get_anno(self, v):
        if v not in self._anno:
            with np.load(self._path(v)) as d:
                self._anno[v] = {k: d[k] for k in d.files}
        return self._anno[v]

    def _get_imgs(self, v):
        if v not in self._imgs:
            folder = os.path.join(self.video_root, v, 'extracted_images')
            f = sorted(glob(os.path.join(folder, '*.jpg'))) or \
                sorted(glob(os.path.join(folder, '*.png')))
            self._imgs[v] = f
        return self._imgs[v]

    def __len__(self):
        return len(self.index)

    # ------------------------------------------------------------------ getitem
    def __getitem__(self, i):
        vi, start = self.index[i]
        v = self.videos[vi]
        anno = self._get_anno(v)
        files = self._get_imgs(v)
        frames = np.arange(start, start + self.seq_len)

        W0, H0 = int(anno['img_size'][0]), int(anno['img_size'][1])
        s = min(self.in_h / H0, self.in_w / W0)
        new_w, new_h = int(round(W0 * s)), int(round(H0 * s))
        pad_x, pad_y = (self.in_w - new_w) // 2, (self.in_h - new_h) // 2

        focal = float(anno['img_focal']) * s
        ic = anno['img_center'].astype(np.float32) * s + np.array([pad_x, pad_y], np.float32)

        color = (np.random.uniform(1 - self.color_scale, 1 + self.color_scale, 3).astype(np.float32)
                 if self.train else np.ones(3, np.float32))

        imgs = []
        for t in frames:
            im = cv2.imread(files[t])[:, :, ::-1]
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((self.in_h, self.in_w, 3), np.float32)
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = im
            if self.train:
                canvas = canvas * color[None, None, :]
            imgs.append(self.normalize_img(np.clip(canvas, 0, 255).astype('uint8')))

        # (2,T,...) -> (T,2,...) so the hand axis sits next to the model's slots
        sl = lambda a: np.ascontiguousarray(a[:, frames].transpose(1, 0, *range(2, a.ndim)))
        j2d = sl(anno['j2d']).astype(np.float32) * s + np.array([pad_x, pad_y], np.float32)
        conf = sl(anno['j2d_conf']).astype(np.float32)
        valid = sl(anno['valid'].astype(np.float32))
        # Joints landing outside the letterboxed frame carry no signal.
        inside = ((j2d[..., 0] >= 0) & (j2d[..., 0] < self.in_w) &
                  (j2d[..., 1] >= 0) & (j2d[..., 1] < self.in_h)).astype(np.float32)

        return {
            'img': torch.stack(imgs).float(),                                    # (T,3,H,W)
            'img_focal': torch.full((self.seq_len,), focal).float(),
            'img_center': torch.from_numpy(ic).float().unsqueeze(0).repeat(self.seq_len, 1),
            'gt_valid': torch.from_numpy(valid).float(),                          # (T,2)
            # Normalized over the input frame, matching how the model projects.
            'gt_j2d': torch.from_numpy(
                j2d / np.array([self.in_w, self.in_h], np.float32) - 0.5).float(),  # (T,2,J,2)
            'gt_j2d_conf': torch.from_numpy(conf * inside).float(),
            'gt_j3d_wo_trans': torch.from_numpy(sl(anno['j3d_wo_trans'])).float(),
            'gt_pose': torch.from_numpy(sl(anno['cam_pose'])).float(),            # (T,2,48)
            'gt_betas': torch.from_numpy(sl(anno['betas'])).float(),              # (T,2,10)
        }
