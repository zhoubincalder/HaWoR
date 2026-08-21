"""
Sapiens2 backbone for HAWOR.

Used frozen: only the 1x1 projection that maps Sapiens2's 2432-d features onto
the 1280-d the space-time module and MANO head expect is trainable, so the rest
of the network is unchanged and the comparison against the ViT-H backbone is
controlled.

Input is the same 256x192 centre slice the ViT path receives, which at patch 16
gives the 16x12 token grid HAWOR.forward_step hardcodes -- no interpolation of
the grid is needed. Note that is far below Sapiens2's 1024x768 pretraining
resolution, which is a real caveat for how much of its capacity is usable here.
"""
import torch
import torch.nn as nn

SAPIENS_HIDDEN = 2432
HAWOR_HIDDEN = 1280
GRID_H, GRID_W = 16, 12


class Sapiens2Wrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        from transformers import Sapiens2Backbone

        model_dir = cfg.MODEL.BACKBONE.get('SAPIENS_DIR', 'weights/sapiens2')
        dtype = torch.bfloat16 if cfg.MODEL.BACKBONE.get('SAPIENS_BF16', True) else torch.float32
        self.backbone = Sapiens2Backbone.from_pretrained(model_dir, dtype=dtype)
        hidden = getattr(self.backbone.config, 'hidden_size', SAPIENS_HIDDEN)
        # Trainable: this is part of the head, not the frozen feature extractor.
        self.proj = nn.Conv2d(hidden, HAWOR_HIDDEN, kernel_size=1)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        nn.init.zeros_(self.proj.bias)
        self._frozen = False
        self._lora = False

    def freeze_pretrained(self):
        """Freeze Sapiens2 itself, leaving the projection trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def enable_lora(self, r=16, alpha=32, dropout=0.05, targets=None,
                    grad_checkpoint=True):
        """Attach LoRA adapters to the (already frozen) trunk.

        A frozen trunk can only be adapted through the 1x1 projection, which is a
        single linear map over its features. LoRA lets the trunk's own attention
        and MLP projections adapt with a small number of extra parameters, at the
        cost of having to backpropagate through the trunk -- so this is markedly
        slower and more memory-hungry than the frozen setting, where the forward
        runs under no_grad.
        """
        from peft import LoraConfig, get_peft_model

        if targets is None:
            targets = ['q_proj', 'k_proj', 'v_proj', 'o_proj',
                       'gate_proj', 'up_proj', 'down_proj']
        cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                         target_modules=list(targets), bias='none')
        if grad_checkpoint:
            # Backpropagating through the trunk stores activations for every
            # layer, which OOMs at useful batch sizes. Recomputing them in the
            # backward pass trades ~30% extra compute for a large memory saving.
            self.backbone.gradient_checkpointing_enable()
            if hasattr(self.backbone, 'enable_input_require_grads'):
                self.backbone.enable_input_require_grads()
        self.backbone = get_peft_model(self.backbone, cfg)
        self._lora = True
        n = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        print(f'LoRA enabled on backbone: r={r} alpha={alpha} '
              f'targets={list(targets)} -> {n/1e6:.1f}M trainable adapter params')

    def train(self, mode: bool = True):
        super().train(mode)
        # With LoRA the adapters train, so the wrapped trunk stays in train mode
        # (Sapiens2 has drop_path_rate 0 and no batchnorm, so that is safe).
        if mode and self._frozen and not self._lora:
            self.backbone.eval()
        return self

    def forward(self, x):
        dtype = getattr(self.backbone, 'dtype', None) or next(self.backbone.parameters()).dtype
        if self._frozen and not self._lora:
            with torch.no_grad():
                feats = self.backbone(x.to(dtype)).feature_maps[-1]
        else:
            feats = self.backbone(x.to(dtype)).feature_maps[-1]
        feats = self.proj(feats.to(self.proj.weight.dtype))
        # At Sapiens2's native 1024x768 the grid is 64x48; HAWOR.forward_step
        # rearranges assuming 16x12, and running the space-time attention over
        # 3072 spatial locations would be intractable anyway. Pool back to 16x12
        # so the rest of the network is identical to the ViT-H path.
        if feats.shape[-2:] != (GRID_H, GRID_W):
            feats = nn.functional.adaptive_avg_pool2d(feats, (GRID_H, GRID_W))
        return feats


def sapiens2(cfg):
    return Sapiens2Wrapper(cfg)
