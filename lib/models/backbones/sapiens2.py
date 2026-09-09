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
        # Only a fully frozen base (LoRA) needs the embedding output forced to
        # require grad for checkpointing. A partial fine-tune does not, and for
        # a train-the-TOP split it is actively harmful: it drags the frozen
        # lower blocks back into the autograd graph, so backward traverses them
        # anyway and the saving disappears.
        self._ckpt_input_grads = True
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
        if self._ckpt_input_grads and hasattr(self.backbone, 'enable_input_require_grads'):
            self.backbone.enable_input_require_grads()
        print(f'Backbone gradient checkpointing enabled '
              f'(input_require_grads={self._ckpt_input_grads}).')

    def _layers(self):
        """The trunk's transformer block list (`model.layer` for Sapiens2)."""
        import torch.nn as nn
        best = None
        for _, mod in self.backbone.named_modules():
            if isinstance(mod, nn.ModuleList) and len(mod) > 8:
                if best is None or len(mod) > len(best):
                    best = mod
        if best is None:
            raise RuntimeError('could not find the transformer layer list')
        return best

    def truncate_layers(self, n):
        """Keep only the first n transformer blocks.

        A smaller trunk than the released checkpoints offer: 24 of the 0.8b's 32
        blocks measures 0.604B, since there is no 0.4b checkpoint to load (a real
        0.4b would need a narrower hidden size, not fewer layers). The kept
        blocks retain their pretrained weights; the discarded ones are the
        deepest, which is the usual place to cut a ViT for a cheaper feature
        extractor.
        """
        layers = self._layers()
        if n >= len(layers):
            return
        del layers[n:]
        cfg = self.backbone.config
        cfg.num_hidden_layers = n
        # The trunk emits feature maps at `out_indices`, which still point at
        # stage 32. Left alone, forward() returns feature_maps=None and dies with
        # "'NoneType' object is not subscriptable" -- the deleted layers are not
        # the problem, the dangling output stage is. Retarget it to the new last
        # block, and trim the per-layer lists that are indexed by depth.
        if getattr(cfg, 'stage_names', None):
            cfg.stage_names = cfg.stage_names[:n + 1]
        cfg.out_indices = [n]
        cfg.out_features = [f'stage{n}']
        kv = getattr(cfg, 'num_key_value_heads_per_layer', None)
        if kv and len(kv) > n:
            cfg.num_key_value_heads_per_layer = kv[:n]
        # The module caches these at construction, so setting the config alone
        # is not enough.
        for attr, val in (('out_indices', cfg.out_indices),
                          ('out_features', cfg.out_features),
                          ('stage_names', getattr(cfg, 'stage_names', None))):
            if val is not None and hasattr(self.backbone, attr):
                setattr(self.backbone, attr, val)
        tot = sum(p.numel() for p in self.backbone.parameters())
        print(f'Backbone truncated to {n} layers ({tot / 1e9:.3f}B params), '
              f'output stage -> {cfg.out_features[0]}.')

    def freeze_above_layer(self, k):
        """Train the first k blocks (plus patch embed); freeze everything above.

        NOTE this is the opposite of the usual recipe. Fine-tuning normally
        adapts the LAST blocks, the ones feeding the task head, and leaves the
        generic early features alone. Training the first k means the frozen
        upper blocks must consume features that are moving underneath them.
        Supported because it was asked for, not because it is the safe default.
        """
        layers = self._layers()
        for p in self.backbone.parameters():
            p.requires_grad = False
        for i in range(min(k, len(layers))):
            for p in layers[i].parameters():
                p.requires_grad = True
        tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.backbone.parameters())
        print(f'Backbone: first {k} of {len(layers)} blocks trainable '
              f'({tr / 1e6:.0f}M of {tot / 1e6:.0f}M params).')
        # Gradients must reach block k-1, so the trunk still needs train mode and
        # recomputation; this is NOT the frozen path.
        self._frozen = False
        self._ckpt_input_grads = False
        self.backbone.train()

    def freeze_below_layer(self, k):
        """Train the LAST k blocks; freeze everything below them.

        The cheap direction, and the conventional one. Backward terminates at the
        first trainable block, so every block below it is forward-only -- unlike
        freeze_above_layer, where gradients must still traverse the frozen upper
        blocks to reach the trainable lower ones and only their weight gradients
        are skipped.
        """
        layers = self._layers()
        n = len(layers)
        k = min(k, n)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for i in range(n - k, n):
            for p in layers[i].parameters():
                p.requires_grad = True
        tr = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        tot = sum(p.numel() for p in self.backbone.parameters())
        print(f'Backbone: last {k} of {n} blocks trainable '
              f'({tr / 1e6:.0f}M of {tot / 1e6:.0f}M params); blocks 0-{n - k - 1} '
              f'forward-only.')
        self._frozen = False
        self._ckpt_input_grads = False
        self.backbone.train()

    def enable_fp8_training(self):
        """Swap the trunk's Linears for torchao Float8Linear (real FP8 training).

        NOT the same thing as MODEL.BACKBONE.FP8, which calls torchao's
        *inference* quantize_() and refuses to run unless the backbone is fully
        frozen, because it replaces weights with non-trainable quantized tensors.
        This keeps master weights in bf16 and runs the matmuls in fp8 with
        dynamic scaling, so gradients still flow -- which is what a partial
        fine-tune needs.

        Requires torch.compile. Measured on sm_120 for one 1280->5120->1280
        block at 3072 tokens: bf16 14.35 ms, fp8 eager 43.75 ms (3x SLOWER, the
        scale/cast ops dominate), fp8 compiled 10.67 ms (1.35x). Enabling this
        without TORCH_COMPILE is a pessimisation, so it says so.
        """
        from torchao.float8 import convert_to_float8_training, Float8LinearConfig
        # fp8 matmuls need both inner dims divisible by 16; skip anything else
        # rather than let torchao fall back silently per-layer.
        def ok(mod, fqn):
            import torch.nn as nn
            return (isinstance(mod, nn.Linear)
                    and mod.in_features % 16 == 0 and mod.out_features % 16 == 0)
        convert_to_float8_training(self.backbone, config=Float8LinearConfig(),
                                   module_filter_fn=ok)
        n = sum(1 for m in self.backbone.modules() if 'Float8' in type(m).__name__)
        print(f'FP8 training enabled on {n} Linear layers (torchao float8).')
        return n

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
