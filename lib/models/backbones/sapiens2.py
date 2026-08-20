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

    def freeze_pretrained(self):
        """Freeze Sapiens2 itself, leaving the projection trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._frozen:
            self.backbone.eval()
        return self

    def forward(self, x):
        if self._frozen:
            with torch.no_grad():
                feats = self.backbone(x.to(self.backbone.dtype)).feature_maps[-1]
        else:
            feats = self.backbone(x.to(self.backbone.dtype)).feature_maps[-1]
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
