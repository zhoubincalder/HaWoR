"""
Full-frame two-hand HAWOR.

Takes whole frames instead of per-hand crops, so no detector or tracker is needed
and both hands come out of one backbone pass. Motivated by Sapiens2, which was
pretrained on full human images at 1024x768 -- feeding it tight hand crops is out
of distribution, which is the likely reason the frozen-Sapiens-on-crops run lost
to the hand-specific ViT-H.

Differences from the crop model in hawor.py:

- Two MANO layers. Mirroring a left hand into right-hand space only works on a
  per-hand crop; here both hands share the image.
- No CLIFF bbox feature. There is no crop box to encode, which also removes the
  box-size depth cue, so translation is parametrised directly: the head predicts
  a normalized image position and a log-depth residual, and the translation is
  back-projected through the known intrinsics. That keeps the 2D loss able to
  drive the hand's image position directly.
- Temporal attention runs over the two hand tokens across time rather than over
  feature-grid locations: on a full frame a hand moves across the grid, so
  per-location attention no longer tracks a hand.
- A per-hand visibility logit, since hands routinely leave the frame.
"""
from typing import Dict

import einops
import numpy as np
import pytorch_lightning as pl
import torch
from yacs.config import CfgNode

from hawor.utils.rotation import angle_axis_to_rotation_matrix
from lib.models.backbones import create_backbone
from lib.models.hawor import load_checkpoint
from lib.models.losses import Keypoint2DLoss, Keypoint3DLoss, ParameterLoss
from lib.models.mano_wrapper import MANO
from lib.models.modules import MANOTwoHandHead, temporal_attention
from lib.utils.geometry import perspective_projection
from lib.utils.geometry import rot6d_to_rotmat_hmr2 as rot6d_to_rotmat

Z0 = 0.5   # metres; typical egocentric hand distance, so cam[2]=0 starts sensibly


class HaworFull(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        self.seq_len = 16
        self.in_h = cfg.MODEL.get('INPUT_H', 384)
        self.in_w = cfg.MODEL.get('INPUT_W', 512)

        self.backbone = create_backbone(cfg)
        self.backbone_frozen = False
        self_pretrained = hasattr(self.backbone, 'freeze_pretrained')
        if cfg.MODEL.BACKBONE.get('PRETRAINED_WEIGHTS', None):
            sd = load_checkpoint(cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS)['state_dict']
            bb = {k[9:]: v for k, v in sd.items() if k.startswith('backbone.')}
            if not bb:
                raise ValueError('checkpoint has no "backbone.*" keys')
            self.backbone.load_state_dict(bb)
            print(f'Loaded backbone from {cfg.MODEL.BACKBONE.PRETRAINED_WEIGHTS}')
        elif self_pretrained:
            print(f'Backbone {cfg.MODEL.BACKBONE.TYPE} loaded its own pretrained weights')
        else:
            print('WARNING: init backbone from scratch !!!')
        if cfg.MODEL.BACKBONE.get('FREEZE', True):
            if self_pretrained:
                self.backbone.freeze_pretrained()
            else:
                for p in self.backbone.parameters():
                    p.requires_grad = False
            self.backbone_frozen = True
            print('Backbone is frozen.')

        # A full fine-tune (FREEZE: False) must recompute activations or it OOMs;
        # enable_lora() handles its own case.
        if (cfg.MODEL.BACKBONE.get('GRAD_CHECKPOINT', True)
                and not cfg.MODEL.BACKBONE.get('LORA', False)
                and not self.backbone_frozen
                and hasattr(self.backbone, 'enable_grad_checkpoint')):
            self.backbone.enable_grad_checkpoint()

        # LoRA adapters on the frozen trunk, so the backbone's features can adapt
        # to full frames rather than only the projection reading them. Rules out
        # fp8: gradients must flow through the trunk, which torchao's quantized
        # weights are not set up for.
        if cfg.MODEL.BACKBONE.get('LORA', False):
            if not self.backbone_frozen:
                raise ValueError('MODEL.BACKBONE.LORA requires FREEZE: True')
            if not hasattr(self.backbone, 'enable_lora'):
                raise ValueError(f'backbone {cfg.MODEL.BACKBONE.TYPE} has no LoRA support')
            self.backbone.enable_lora(
                r=cfg.MODEL.BACKBONE.get('LORA_R', 16),
                alpha=cfg.MODEL.BACKBONE.get('LORA_ALPHA', 32),
                dropout=cfg.MODEL.BACKBONE.get('LORA_DROPOUT', 0.05),
                targets=cfg.MODEL.BACKBONE.get('LORA_TARGETS', None),
                grad_checkpoint=cfg.MODEL.BACKBONE.get('GRAD_CHECKPOINT', True))
            if cfg.MODEL.BACKBONE.get('FP8', False):
                print('NOTE: FP8 skipped because LoRA needs gradients through the trunk.')
        elif cfg.MODEL.BACKBONE.get('FP8', False):
            from lib.models.hawor import HAWOR
            HAWOR._quantize_backbone_fp8(self)

        self.head = MANOTwoHandHead(cfg, context_dim=1280, use_init_cam=False)

        # Temporal attention over each hand's token sequence.
        if cfg.MODEL.get('MOTION_MODULE', True):
            self.motion_module = temporal_attention(
                in_dim=1024, out_dim=1024,
                hdim=cfg.MODEL.get('MOTION_HDIM', 512),
                nlayer=cfg.MODEL.get('MOTION_NLAYER', 6), residual=True)
            print('Using temporal attention over hand tokens.')
        else:
            self.motion_module = None

        reduction = cfg.TRAIN.get('LOSS_REDUCTION', 'mean')
        self.keypoint_3d_loss = Keypoint3DLoss('l1', reduction=reduction)
        self.keypoint_2d_loss = Keypoint2DLoss('l1', reduction=reduction)
        self.mano_parameter_loss = ParameterLoss(reduction=reduction)
        self.vis_loss = torch.nn.BCEWithLogitsLoss()

        mano_cfg = {k.lower(): v for k, v in dict(cfg.MANO).items()}
        self.mano_right = MANO(**mano_cfg, is_rhand=True)
        left_cfg = dict(mano_cfg)
        left_cfg['model_path'] = cfg.MANO.get('MODEL_PATH_LEFT', mano_cfg['model_path'])
        self.mano_left = MANO(**left_cfg, is_rhand=False)
        # Known MANO left-hand shapedirs sign bug (smplx issue #48).
        if torch.sum(torch.abs(self.mano_left.shapedirs[:, 0, :]
                               - self.mano_right.shapedirs[:, 0, :])) < 1:
            self.mano_left.shapedirs[:, 0, :] *= -1

        self.automatic_optimization = False

        # Warm start from a previous run's weights, tolerating structural
        # differences. Adding LoRA wraps the backbone with peft, which renames its
        # keys, so a strict load (or Lightning's --resume) fails -- but the head,
        # temporal module and projection are unchanged and worth keeping rather
        # than retraining from scratch.
        ws = cfg.MODEL.get('WARM_START', None)
        if ws:
            sd = load_checkpoint(ws)['state_dict']
            missing, unexpected = self.load_state_dict(sd, strict=False)
            loaded = len(sd) - len(unexpected)
            print(f'Warm start from {ws}: loaded {loaded}/{len(sd)} tensors '
                  f'({len(missing)} missing, {len(unexpected)} unexpected)')
            head_missing = [k for k in missing if not k.startswith('backbone.')]
            if head_missing:
                print(f'  NOTE {len(head_missing)} non-backbone keys did not load, '
                      f'e.g. {head_missing[:3]}')

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.backbone_frozen and not hasattr(self.backbone, 'freeze_pretrained'):
            self.backbone.eval()
        return self

    def get_parameters(self):
        p = list(self.head.parameters()) + list(self.backbone.parameters())
        if self.motion_module is not None:
            p += list(self.motion_module.parameters())
        return p

    def configure_optimizers(self):
        return torch.optim.AdamW(
            [{'params': filter(lambda q: q.requires_grad, self.get_parameters()),
              'lr': self.cfg.TRAIN.LR}],
            weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)

    def cam_to_trans(self, cam, focal, center):
        """(u_norm, v_norm, log_depth) -> camera-space translation, via the known
        intrinsics. Predicting image position rather than a crop-relative offset
        keeps the 2D loss directly informative about where the hand is."""
        u = (cam[..., 0] + 0.5) * self.in_w
        v = (cam[..., 1] + 0.5) * self.in_h
        z = Z0 * torch.exp(cam[..., 2].clamp(-2.0, 2.0))
        tx = (u - center[..., 0]) * z / focal
        ty = (v - center[..., 1]) * z / focal
        return torch.stack([tx, ty, z], dim=-1)

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        img = batch['img']                                    # (B,T,3,H,W)
        B, T = img.shape[:2]
        feat = self.backbone(img.flatten(0, 1)).float()        # (B*T,C,h,w)

        tok = self.head(feat, return_tokens=True)              # (B*T,2,1024)
        if self.motion_module is not None:
            tok = einops.rearrange(tok, '(b t) n c -> (b n) t c', t=T)
            tok = self.motion_module(tok)
            tok = einops.rearrange(tok, '(b n) t c -> (b t) n c', n=2)
        pose, shape, cam, vis = self.head.decode(tok)          # (B*T,2,·)

        rotmat = rot6d_to_rotmat(pose.reshape(-1, 6)).reshape(B * T, 2, 16, 3, 3)
        focal = batch['img_focal'].flatten(0, 1)               # (B*T,)
        center = batch['img_center'].flatten(0, 1)             # (B*T,2)
        # broadcast over the two hand slots for the translation decode
        trans = self.cam_to_trans(cam, focal[:, None], center[:, None, :])   # (B*T,2,3)

        j3d, j2d = [], []
        for slot, mano in ((0, self.mano_left), (1, self.mano_right)):
            out = mano(global_orient=rotmat[:, slot, :1],
                       hand_pose=rotmat[:, slot, 1:],
                       betas=shape[:, slot], pose2rot=False)
            j = out.joints
            j3d.append(j)
            # Anchor the hand at its WRIST before adding the predicted
            # translation. MANO rotates about its rest-pose root joint J0, so
            # with transl=0 the wrist sits at J0 -- 96.1mm from the origin, and
            # at OPPOSITE signs in x for the two hands (+-0.0957, 0.0064,
            # 0.0062). Without this subtraction the wrist lands at J0 + trans
            # while cam_to_trans builds trans so that (u, v) projects `trans`
            # exactly, so the predicted image position would refer to a point
            # 96mm away from the hand -- by ~121px at z=0.5m, and z-dependently
            # (203px at 0.3m, 76px at 0.8m). The network can learn to absorb
            # that, which is why trained models were not wrong, but it has to
            # learn a depth- and slot-dependent offset to do it. Subtracting the
            # wrist makes (u, v) mean what cam_to_trans's docstring says it
            # means: where the hand is in the image.
            pts = (j - j[:, :1]) + trans[:, slot][:, None]
            px = perspective_projection(pts, rotation=None, translation=None,
                                        focal_length=focal, camera_center=center)
            j2d.append(px / torch.tensor([self.in_w, self.in_h],
                                         device=px.device, dtype=px.dtype) - 0.5)
        return {
            'pred_pose': pose, 'pred_shape': shape, 'pred_cam': cam, 'pred_vis': vis,
            'pred_rotmat': rotmat, 'pred_trans': trans,
            'pred_keypoints_3d': torch.stack(j3d, dim=1),      # (B*T,2,J,3)
            'pred_keypoints_2d': torch.stack(j2d, dim=1),      # (B*T,2,J,2)
        }

    def compute_loss(self, batch: Dict, out: Dict, train: bool = True) -> torch.Tensor:
        w = self.cfg.LOSS_WEIGHTS
        valid = batch['gt_valid'].flatten(0, 1)                # (B*T,2)
        m = valid.reshape(-1) > 0                              # per (sample,hand)

        gt2 = batch['gt_j2d'].flatten(0, 1).flatten(0, 1)      # (B*T*2,J,2)
        c2 = batch['gt_j2d_conf'].flatten(0, 1).flatten(0, 1).unsqueeze(-1)
        p2 = out['pred_keypoints_2d'].flatten(0, 1)
        gt3 = batch['gt_j3d_wo_trans'].flatten(0, 1).flatten(0, 1)
        p3 = out['pred_keypoints_3d'].flatten(0, 1)
        pose_gt = angle_axis_to_rotation_matrix(
            batch['gt_pose'].flatten(0, 1).flatten(0, 1).reshape(-1, 3)).reshape(-1, 16, 3, 3)
        betas_gt = batch['gt_betas'].flatten(0, 1).flatten(0, 1)

        # Absent hands must not be supervised: drop them rather than weighting
        # them, so they contribute nothing to the mean.
        if m.any():
            l2d = self.keypoint_2d_loss(p2[m], torch.cat([gt2[m], c2[m]], -1))
            l3d = self.keypoint_3d_loss(
                p3[m], torch.cat([gt3[m], torch.ones_like(gt3[m][..., :1])], -1), pelvis_id=0)
            rot = out['pred_rotmat'].flatten(0, 1)
            lo = self.mano_parameter_loss(rot[m][:, :1].reshape(m.sum(), -1),
                                          pose_gt[m][:, :1].reshape(m.sum(), -1))
            lh = self.mano_parameter_loss(rot[m][:, 1:].reshape(m.sum(), -1),
                                          pose_gt[m][:, 1:].reshape(m.sum(), -1))
            lb = self.mano_parameter_loss(out['pred_shape'].flatten(0, 1)[m],
                                          betas_gt[m])
        else:
            z = p2.sum() * 0
            l2d = l3d = lo = lh = lb = z

        lvis = self.vis_loss(out['pred_vis'].reshape(-1), valid.reshape(-1))
        loss = (w['KEYPOINTS_2D'] * torch.nan_to_num(l2d) + w['KEYPOINTS_3D'] * l3d
                + w['GLOBAL_ORIENT'] * lo + w['HAND_POSE'] * lh + w['BETAS'] * lb
                + self.cfg.LOSS_WEIGHTS.get('VISIBILITY', 0.01) * lvis)
        out['losses'] = {'loss': loss.detach(), 'loss_keypoints_2d': l2d.detach(),
                         'loss_keypoints_3d': l3d.detach(), 'loss_vis': lvis.detach()}
        return loss

    def training_step(self, batch, batch_idx):
        batch = batch['img'] if 'img' in batch and isinstance(batch['img'], dict) else batch
        opt = self.optimizers(use_pl_optimizer=True)
        out = self.forward_step(batch, train=True)
        loss = self.compute_loss(batch, out, train=True)
        if torch.isnan(loss):
            raise ValueError('Loss is NaN')
        opt.zero_grad()
        self.manual_backward(loss)
        if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
            gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(),
                                                self.cfg.TRAIN.GRAD_CLIP_VAL)
            self.log('train/grad_norm', gn, on_step=True, prog_bar=True,
                     batch_size=batch['img'].shape[0])
        opt.step()
        self.log('train/loss', out['losses']['loss'], on_step=True, prog_bar=True,
                 batch_size=batch['img'].shape[0])
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, out, train=False)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch['img'].shape[0])
        return out
