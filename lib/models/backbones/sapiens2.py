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
        # from_pretrained() returns the trunk in eval mode. nn.Module's default is
        # train mode, so leaving it as-is makes the trunk disagree with its parent
        # for any run where nobody calls .train() -- and Lightning does not call it
        # before the first training batch. That silently disables gradient
        # checkpointing, because transformers guards recomputation on
        # `self.gradient_checkpointing and self.training`: the flag was set, the
        # trunk was in eval, and zero blocks were ever recomputed. Cost was ~6x
        # peak memory (44.4 GB vs 10.8 GB at 384x512), which is what forced the
        # small batch sizes in the LoRA runs. freeze_pretrained() re-evals it.
        self.backbone.train()
        hidden = getattr(self.backbone.config, 'hidden_size', SAPIENS_HIDDEN)
        # Trainable: this is part of the head, not the frozen feature extractor.
        self.proj = nn.Conv2d(hidden, HAWOR_HIDDEN, kernel_size=1)
        nn.init.xavier_uniform_(self.proj.weight, gain=0.01)
        nn.init.zeros_(self.proj.bias)
        self._frozen = False
        self._lora = False
        pg = cfg.MODEL.BACKBONE.get('POOL_GRID', None)
        self.pool_grid = tuple(pg) if pg else None
        if self.pool_grid:
            print(f'Backbone features pooled to {self.pool_grid[0]}x{self.pool_grid[1]}.')

    def enable_grad_checkpoint(self):
        """Recompute activations in the backward pass instead of storing them.

        Required for any configuration that backpropagates through the trunk --
        full fine-tuning at 768x1024 needs ~15GB with this on and OOMs above
        99GB with it off. enable_lora() turns it on itself; a full fine-tune
        (FREEZE: False) has to ask for it.
        """
        self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, 'enable_input_require_grads'):
            self.backbone.enable_input_require_grads()
        print('Backbone gradient checkpointing enabled.')

    def freeze_pretrained(self):
        """Freeze Sapiens2 itself, leaving the projection trainable."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        self._frozen = True
        # train() re-applies this, but set it here too: nothing guarantees train()
        # is ever called (see __init__), and a frozen trunk belongs in eval.
        self.backbone.eval()

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
        # Pool only when asked. The crop model's space-time module rearranges with
        # a hardcoded 16x12 grid, so that path needs POOL_GRID; the full-frame
        # model's head flattens the grid into a cross-attention context and works
        # at any size, so it keeps the backbone's native resolution -- at
        # 768x1024 that is 48x64 instead of 16x12, a 16x finer spatial grid, and
        # nearly free because only two query tokens attend over it.
        if self.pool_grid is not None and tuple(feats.shape[-2:]) != self.pool_grid:
            feats = nn.functional.adaptive_avg_pool2d(feats, self.pool_grid)
        return feats


def sapiens2(cfg):
    return Sapiens2Wrapper(cfg)
