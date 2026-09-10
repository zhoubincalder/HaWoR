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
import math
from contextlib import nullcontext
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
# Reference focal for the depth decode, in pixels: the corpus median effective
# focal (dexycb/ho3d/h2o3d all sit at ~615-617). Depth is decoded as
# Z0 * (f/F_REF) * exp(dz), so at f = F_REF the formula is exactly the old one
# and those three datasets are unchanged.
F_REF = 615.0


class HaworFull(pl.LightningModule):

    def __init__(self, cfg: CfgNode):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg
        # Frames per window. The temporal module and forward_step both take T
        # from the tensor, so this only has to agree with the dataset. Halving it
        # halves tokens per step, which is the cheapest way to get under the
        # memory ceiling that forces gradient checkpointing.
        self.seq_len = cfg.MODEL.get('SEQ_LEN', 16)
        self.in_h = cfg.MODEL.get('INPUT_H', 384)
        self.in_w = cfg.MODEL.get('INPUT_W', 512)
        # 0 = recompute every step (the plain GRAD_CHECKPOINT behaviour).
        self._ckpt_above = int(cfg.MODEL.BACKBONE.get('CKPT_ABOVE_TOKENS', 0))
        self._loss_hist = []
        self._loss_hist_n = max(1, int(cfg.TRAIN.get('RUNNING_MEAN_WINDOW', 200)))

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
        # Optional smaller trunk: keep only the first N blocks. Applied before
        # any freezing so the two compose.
        nl = cfg.MODEL.BACKBONE.get('NUM_LAYERS', 0)
        if nl and hasattr(self.backbone, 'truncate_layers'):
            self.backbone.truncate_layers(int(nl))

        # Pool the patch grid 2x2 mid-trunk, so the deep blocks run at a quarter
        # of the tokens. Not the same as POOL_GRID, which pools the trunk's
        # output and saves nothing.
        mp = int(cfg.MODEL.BACKBONE.get('MIDTRUNK_POOL_AFTER', 0))
        if mp and hasattr(self.backbone, 'enable_midtrunk_pool'):
            self.backbone.enable_midtrunk_pool(mp)

        # Partial fine-tune: train the first K blocks, freeze the rest. Distinct
        # from FREEZE (nothing trains) and from LORA (adapters everywhere).
        # Positive K trains the FIRST K blocks; negative trains the LAST |K|.
        # The sign matters for speed, not just for which weights move: with the
        # last blocks trainable, backward stops at the first trainable block and
        # everything below is forward-only. With the first blocks trainable,
        # gradients still traverse every frozen block above to reach them.
        tl = int(cfg.MODEL.BACKBONE.get('TRAINABLE_LAYERS', 0))
        if tl > 0 and hasattr(self.backbone, 'freeze_above_layer'):
            self.backbone.freeze_above_layer(tl)
        elif tl < 0 and hasattr(self.backbone, 'freeze_below_layer'):
            self.backbone.freeze_below_layer(-tl)

        if not tl and cfg.MODEL.BACKBONE.get('FREEZE', True):
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
        elif cfg.MODEL.BACKBONE.get('FP8_TRAINING', False):
            # Real fp8 TRAINING (gradients flow), unlike FP8 which is inference
            # quantization and skips itself unless the backbone is fully frozen.
            if not cfg.MODEL.BACKBONE.get('TORCH_COMPILE', 0):
                print('WARNING: FP8_TRAINING without TORCH_COMPILE is ~3x SLOWER '
                      'than bf16; the cast/scale ops need fusing.')
            self.backbone.enable_fp8_training()
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

        # AFTER the warm start, not before: torch.compile returns an
        # OptimizedModule that renames every child key with an `_orig_mod.`
        # prefix, so a state_dict saved from an uncompiled run would land
        # entirely in `unexpected` and, since this loads with strict=False,
        # would leave the backbone at its pretrained weights without failing.
        # Applied to the backbone and head only -- not the losses, which mask by
        # per-frame validity and so have data-dependent shapes that would force
        # a recompile every step.
        if cfg.MODEL.BACKBONE.get('TORCH_COMPILE', 0):
            mode = cfg.MODEL.BACKBONE.get('TORCH_COMPILE_MODE', 'default')
            # Dynamo compiles one graph per input shape and gives up after
            # `recompile_limit` of them (default 8), silently running every
            # later shape in EAGER for the rest of the run. Under NATIVE_RES the
            # shape count is (source sizes) x (jitter levels) -- 12 here -- so
            # the default is exceeded at about step 25 and a third of the shapes
            # never get compiled. That costs speed and, since compile also cuts
            # activation memory (59.2 -> 41.1 GB at 1200 tokens), memory too.
            import torch._dynamo as _dynamo
            lim = int(cfg.MODEL.BACKBONE.get('RECOMPILE_LIMIT', 64))
            for attr in ('recompile_limit', 'cache_size_limit'):   # renamed in 2.x
                if hasattr(_dynamo.config, attr):
                    setattr(_dynamo.config, attr, lim)
            self.backbone = torch.compile(self.backbone, mode=mode)
            self.head = torch.compile(self.head, mode=mode)
            print(f'torch.compile enabled on backbone and head '
                  f'(mode={mode}, recompile_limit={lim})')

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
        """AdamW with a lower LR on the pretrained trunk, linear warmup, cosine decay.

        Three departures from the released recipe's single flat 1e-5, all aimed
        at the same thing: a 50M-parameter head starting from random init sends
        large, uninformative gradients into an 815M pretrained trunk from step
        one. Measured on the first run without them, the gradient norm sat at
        ~9 (spiking to 16.4) for 4000 steps with no decay, every update was
        clipped, and the running-mean loss plateaued at ~0.176 by step 3100.

          * TRAIN.BACKBONE_LR_MULT scales the trunk's LR relative to the head's,
            so the pretrained features are not overwritten while the head is
            still noise.
          * TRAIN.WARMUP_STEPS ramps linearly from zero, which is the standard
            remedy for exactly those large early gradients.
          * cosine decay to TRAIN.MIN_LR_RATIO of peak over TRAIN.COSINE_STEPS
            optimizer steps (0 = derive from the trainer's estimate).

        LambdaLR multiplies each group's own base_lr, so warmup and decay apply
        to both groups while preserving the ratio between them.
        """
        base = float(self.cfg.TRAIN.LR)
        mult = float(self.cfg.TRAIN.get('BACKBONE_LR_MULT', 1.0))
        # self.backbone may be an OptimizedModule by now; .parameters() forwards
        # to the wrapped module, so these are the same tensors either way.
        trunk = [q for q in self.backbone.parameters() if q.requires_grad]
        trunk_ids = {id(q) for q in trunk}
        rest = [q for q in self.get_parameters()
                if q.requires_grad and id(q) not in trunk_ids]
        groups = [{'params': trunk, 'lr': base * mult},
                  {'params': rest, 'lr': base}]
        opt = torch.optim.AdamW(groups, weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        print(f'AdamW: backbone lr {base * mult:.2e} ({len(trunk)} tensors), '
              f'head lr {base:.2e} ({len(rest)} tensors)')

        warm = max(0, int(self.cfg.TRAIN.get('WARMUP_STEPS', 0)))
        total = int(self.cfg.TRAIN.get('COSINE_STEPS', 0))
        if total <= 0:
            # estimated_stepping_batches counts BATCHES here, because this
            # module drives accumulation itself rather than through
            # Trainer(accumulate_grad_batches=...), which it ignores.
            n = max(1, int(self.cfg.TRAIN.get('ACCUM_STEPS', 1)))
            try:
                total = max(1, int(self.trainer.estimated_stepping_batches) // n)
            except Exception:
                total = warm + 1
        floor = float(self.cfg.TRAIN.get('MIN_LR_RATIO', 0.05))
        print(f'LR schedule: {warm} warmup steps, cosine to {floor:.0%} of peak '
              f'over {total} optimizer steps')

        def lr_lambda(step):
            if warm and step < warm:
                return (step + 1) / warm
            span = max(1, total - warm)
            prog = min(1.0, max(0.0, (step - warm) / span))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

        sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {'optimizer': opt,
                'lr_scheduler': {'scheduler': sch, 'interval': 'step'}}

    def cam_to_trans(self, cam, focal, center, size=None):
        """(u_norm, v_norm, log_depth) -> camera-space translation, via the known
        intrinsics. Predicting image position rather than a crop-relative offset
        keeps the 2D loss directly informative about where the hand is.

        `size` is the per-sample (W, H) of the input tensor, broadcastable against
        cam. Under MODEL.NATIVE_RES every sample carries its own input size, so
        taking it from the config would denormalize (u, v) against the wrong
        frame -- silently, and by up to 1.6x on the 640x480 datasets.

        Depth carries the focal length. Apparent size determines depth only up
        to f: a hand of metric size S at depth z spans about f*S/z pixels, so
        z = f*S/pixels. The network sees pixels and never sees f -- img_focal is
        read here, not fed to the backbone -- so decoding z from dz alone asks
        it to guess a per-camera constant. Measured on the step-9188 checkpoint,
        root-depth error tracked effective focal at r = 0.859 across the six val
        folds (hot3d f=270 -> 165mm, arctic f=633 -> 524mm), and the hot3d /
        dexycb pair pins it exactly: their true depths differ 3.3x while their
        apparent hand sizes differ only 1.43x, and the residual 2.30 equals
        f_dexycb/f_hot3d = 2.28 to within 1%. Scaling by f/F_REF makes dz a
        focal-INDEPENDENT quantity (essentially S/pixels) and lets the geometry
        supply the rest, which also makes the model valid on an unseen camera."""
        if size is None:
            in_w, in_h = self.in_w, self.in_h
        else:
            in_w, in_h = size[..., 0], size[..., 1]
        u = (cam[..., 0] + 0.5) * in_w
        v = (cam[..., 1] + 0.5) * in_h
        z = Z0 * (focal / F_REF) * torch.exp(cam[..., 2].clamp(-2.0, 2.0))
        tx = (u - center[..., 0]) * z / focal
        ty = (v - center[..., 1]) * z / focal
        return torch.stack([tx, ty, z], dim=-1)

    def forward_step(self, batch: Dict, train: bool = False) -> Dict:
        img = batch['img']                                    # (B,T,3,H,W)
        B, T = img.shape[:2]
        # Recompute activations only for the windows that would not otherwise
        # fit. Peak memory is linear in tokens (~0.04GB/token plus ~10GB of
        # weights and optimizer state), so under NATIVE_RES most windows are
        # small enough to keep their activations and skip the ~30% recompute
        # cost, while the largest few still need it.
        if self._ckpt_above and self.training:
            tok = (img.shape[-2] // 16) * (img.shape[-1] // 16)
            self.backbone.set_checkpointing(tok > self._ckpt_above)
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
        if 'img_size' in batch:
            size = batch['img_size'].flatten(0, 1)             # (B*T,2) as (W,H)
        else:
            size = img.new_tensor([self.in_w, self.in_h]).expand(B * T, 2)
        # broadcast over the two hand slots for the translation decode
        trans = self.cam_to_trans(cam, focal[:, None], center[:, None, :],
                                  size[:, None, :])            # (B*T,2,3)

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
            j2d.append(px / size[:, None, :].to(px.dtype) - 0.5)
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

        # Gradient accumulation, implemented here because this module sets
        # automatic_optimization = False -- under manual optimization Lightning
        # IGNORES Trainer(accumulate_grad_batches=...), so setting that flag
        # would silently do nothing.
        #
        # Why it matters: peak memory tracks frames per step, and 16 frames is
        # the ceiling without gradient checkpointing. Accumulation reaches the
        # released recipe's 64-frame update at 16-frame memory, which is what
        # makes the fp8/no-checkpoint path usable rather than merely fast at a
        # batch too small to train with.
        n = max(1, int(self.cfg.TRAIN.get('ACCUM_STEPS', 1)))
        first = (batch_idx % n) == 0
        last = ((batch_idx + 1) % n) == 0

        if first:
            opt.zero_grad()
        # LOSS_REDUCTION is 'mean', so each micro-batch returns a mean over its
        # own frames. Dividing by n makes the accumulated gradient the mean over
        # the whole effective batch rather than n times it -- without this the
        # effective learning rate scales with ACCUM_STEPS.
        # Under DDP every backward triggers an all-reduce of all 864M gradients
        # (~1.7GB in bf16). Accumulation only pays off if the non-final
        # micro-steps skip that: otherwise N micro-steps cost N all-reduces and
        # accumulation ADDS communication instead of amortizing it. The DGX has
        # no NVLink -- nvidia-smi reports "Device does not have or support
        # Nvlink" and the topology is all NODE, i.e. PCIe through a host bridge
        # -- so the collective is expensive enough to matter: per-rank window
        # time there is 0.754s against 0.535s for the same GPU running alone.
        sync = nullcontext()
        if not last:
            ddp = getattr(self.trainer, 'model', None)
            if ddp is not None and hasattr(ddp, 'no_sync'):
                sync = ddp.no_sync()
        with sync:
            self.manual_backward(loss / n)

        if last:
            if self.cfg.TRAIN.get('GRAD_CLIP_VAL', 0) > 0:
                # Clip the ACCUMULATED gradient, once per update. Clipping each
                # micro-step would clip partial gradients and change the
                # direction of the update, not just its norm.
                gn = torch.nn.utils.clip_grad_norm_(self.get_parameters(),
                                                    self.cfg.TRAIN.GRAD_CLIP_VAL)
                self.log('train/grad_norm', gn, on_step=True, prog_bar=True,
                         batch_size=batch['img'].shape[0])
            opt.step()
            # Manual optimization: Lightning does NOT step schedulers for us,
            # and it must advance once per OPTIMIZER step, not once per batch.
            sch = self.lr_schedulers()
            if sch is not None:
                sch.step()
                lrs = [g['lr'] for g in opt.param_groups]
                self.log('lr/backbone', lrs[0], on_step=True, batch_size=1)
                if len(lrs) > 1:
                    self.log('lr/head', lrs[-1], on_step=True, batch_size=1)
        bs = batch['img'].shape[0]
        raw = out['losses']['loss']
        self.log('train/loss', raw, on_step=True, prog_bar=True, batch_size=bs)

        # A single-window loss is dominated by WHICH dataset the batch drew, so
        # the raw curve is close to unreadable at BATCH_SIZE 1. Two additions
        # make it diagnostic: a running mean over the last RUNNING_MEAN_WINDOW
        # micro-steps, which averages the dataset mixture out, and one scalar
        # per dataset, which removes the mixture entirely.
        self._loss_hist.append(float(raw))
        if len(self._loss_hist) > self._loss_hist_n:
            self._loss_hist.pop(0)
        self.log('train/loss_mean', sum(self._loss_hist) / len(self._loss_hist),
                 on_step=True, prog_bar=True, batch_size=bs)

        name = batch.get('ds_name')
        if name is not None:
            # default_collate turns the per-sample string into a list.
            name = name[0] if isinstance(name, (list, tuple)) else name
            self.log(f'train_ds/{name}', raw, on_step=True, prog_bar=False,
                     batch_size=bs)
        return out

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        out = self.forward_step(batch, train=False)
        loss = self.compute_loss(batch, out, train=False)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True,
                 batch_size=batch['img'].shape[0])
        return out
