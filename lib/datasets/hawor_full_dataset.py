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
- MODEL.NATIVE_RES switches the target from a fixed canvas to each source's own
  resolution, padded up to a multiple of the 16px patch. Nothing is upsampled
  (the corpus is 640x480 to 840x600, all smaller than 1024x768, so the fixed
  canvas was interpolating 93.8% of frames up for no added information) and the
  letterbox dead space goes away with it. A source too large for the canvas is
  still scaled down to fit, so the token budget is never exceeded. Windows then
  vary in size between samples, which is fine at BATCH_SIZE 1 -- a window's 16
  frames all come from one sequence, so they always agree -- but needs
  aspect-ratio bucketing above that, since collate stacks (B,T,3,H,W).
- Ground truth keeps a leading hand axis (slot 0 left, slot 1 right) plus a
  per-hand validity flag, since on a full frame a hand is often absent -- with
  crops that case never reached the model.
- Only colour augmentation, plus resolution jitter under NATIVE_RES. Scale and
  translation jitter existed to perturb the crop box; there is no crop here, and
  warping the frame would change the intrinsics. Resolution jitter is the one
  scale augmentation that survives, because a uniform rescale is exactly what
  the letterbox already does to the camera. It only ever scales DOWN -- scaling
  up would invent pixels, which is what native mode exists to avoid -- and it
  draws from a short discrete list rather than a continuous range so the number
  of distinct tensor shapes stays bounded for torch.compile.
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
        self.native_res = bool(cfg.MODEL.get('NATIVE_RES', False))
        self.jitter_levels = list(cfg.MODEL.get('RES_JITTER_LEVELS', []))
        self.max_tokens = int(cfg.MODEL.get('MAX_TOKENS', 0))
        self._budget = {}
        # Which export this sample came from. train_full.py concatenates six
        # roots whose loss scales span more than 20x (arctic 0.044 to dexycb
        # 0.435), so at BATCH_SIZE 1 a single 'train/loss' curve mostly reports
        # which dataset the batch drew, not whether the model improved.
        self.ds_name = os.path.basename(os.path.normpath(video_root)).replace('_export', '')
        if self.native_res and cfg.TRAIN.get('BATCH_SIZE', 1) > 1:
            # Fail here rather than in default_collate, which reports only a
            # shape mismatch and does not say why the shapes differ.
            raise ValueError(
                f'MODEL.NATIVE_RES needs TRAIN.BATCH_SIZE 1, got '
                f'{cfg.TRAIN.BATCH_SIZE}: windows differ in size between '
                f'datasets and between resolution-jitter draws, and collate '
                f'stacks them into one (B,T,3,H,W) tensor. Use ACCUM_STEPS for '
                f'a larger effective batch, or add aspect-ratio bucketing.')
        self.color_scale = cfg.DATASETS.CONFIG.get('COLOR_SCALE', 0.2)

        self.normalize_img = Compose([
            ToTensor(),
            Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD),
        ])
        with open(os.path.join(video_root, set_file)) as f:
            self.videos = json.load(f)
        self._anno, self._imgs = {}, {}
        self.index = self._build_index(stride)
        size = ('native, capped at %dx%d' % (self.in_h, self.in_w) if self.native_res
                else '%dx%d' % (self.in_h, self.in_w))
        jit = (f', jitter {self.jitter_levels}'
               if self.native_res and self.jitter_levels and train else '')
        print(f'[HaworFullDataset] {len(self.videos)} sequences -> {len(self.index)} '
              f'windows of {seq_len} frames at {size}{jit} (train={train})')

    def _budget_scale(self, W0, H0):
        """Largest scale <= 1 whose padded patch grid fits MODEL.MAX_TOKENS.

        Peak memory is linear in tokens, so a token budget is the honest way to
        express the memory ceiling: it caps every dataset at once, needs no
        per-dataset tuning, and covers any dataset added later. Capping the
        input instead of recomputing activations is a real trade -- ARCTIC and
        H2O give up their top resolution, and with it some hand detail -- but it
        means NO window exceeds the budget, so recomputation can be switched off
        for the whole run rather than for most of it.

        Solved by search rather than in closed form because the padding is a
        ceiling to a multiple of 16, which is a step function.
        """
        key = (W0, H0)
        if key in self._budget:
            return self._budget[key]
        tok = lambda s: (-(-int(round(H0 * s)) // 16)) * (-(-int(round(W0 * s)) // 16))
        s = 1.0
        while s > 0.05 and tok(s) > self.max_tokens:
            s -= 0.005
        self._budget[key] = s
        return s

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
        fit = min(self.in_h / H0, self.in_w / W0)
        if self.native_res:
            # min(1, fit): keep the source's own pixels, and scale down only when
            # it does not fit the canvas budget. Then pad TIGHTLY to a multiple of
            # the patch -- 840x600 -> 848x608, not 1024x768 -- because a conv with
            # stride 16 silently DROPS the remainder rather than erroring.
            s = min(1.0, fit)
            # Budget cap BEFORE jitter, so jitter still scales down from the
            # largest allowed size and keeps its full spread.
            if self.max_tokens:
                s = min(s, self._budget_scale(W0, H0))
            if self.train and self.jitter_levels:
                s *= float(np.random.choice(self.jitter_levels))
            new_w, new_h = int(round(W0 * s)), int(round(H0 * s))
            out_w, out_h = -(-new_w // 16) * 16, -(-new_h // 16) * 16
        else:
            s = fit
            new_w, new_h = int(round(W0 * s)), int(round(H0 * s))
            out_w, out_h = self.in_w, self.in_h
        pad_x, pad_y = (out_w - new_w) // 2, (out_h - new_h) // 2

        focal = float(anno['img_focal']) * s
        ic = anno['img_center'].astype(np.float32) * s + np.array([pad_x, pad_y], np.float32)

        color = (np.random.uniform(1 - self.color_scale, 1 + self.color_scale, 3).astype(np.float32)
                 if self.train else np.ones(3, np.float32))

        imgs = []
        for t in frames:
            im = cv2.imread(files[t])[:, :, ::-1]
            im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((out_h, out_w, 3), np.float32)
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
        inside = ((j2d[..., 0] >= 0) & (j2d[..., 0] < out_w) &
                  (j2d[..., 1] >= 0) & (j2d[..., 1] < out_h)).astype(np.float32)

        return {
            'ds_name': self.ds_name,
            'img': torch.stack(imgs).float(),                                    # (T,3,H,W)
            'img_focal': torch.full((self.seq_len,), focal).float(),
            'img_center': torch.from_numpy(ic).float().unsqueeze(0).repeat(self.seq_len, 1),
            'gt_valid': torch.from_numpy(valid).float(),                          # (T,2)
            # Normalized over the input frame, matching how the model projects.
            # The model reads (W,H) from here rather than from the config, so the
            # (u,v) decode and the 2D loss follow a per-sample input size.
            'img_size': torch.tensor([out_w, out_h]).float().unsqueeze(0).repeat(self.seq_len, 1),
            'gt_j2d': torch.from_numpy(
                j2d / np.array([out_w, out_h], np.float32) - 0.5).float(),  # (T,2,J,2)
            'gt_j2d_conf': torch.from_numpy(conf * inside).float(),
            'gt_j3d_wo_trans': torch.from_numpy(sl(anno['j3d_wo_trans'])).float(),
            'gt_pose': torch.from_numpy(sl(anno['cam_pose'])).float(),            # (T,2,48)
            'gt_betas': torch.from_numpy(sl(anno['betas'])).float(),              # (T,2,10)
        }
