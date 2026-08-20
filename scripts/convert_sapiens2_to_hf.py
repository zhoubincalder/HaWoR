"""
Convert an original-format Sapiens2 checkpoint to transformers' Sapiens2Backbone layout.

facebook/sapiens2-* ship weights in the original Sapiens naming
(`backbone.blocks.N.attn.wq...`), not the transformers naming
(`model.layer.N.attention.q_proj...`), and transformers has no conversion
mapping for them. Loading one directly leaves EVERY parameter randomly
initialized while reporting only a warning, which silently produces a
random 5B backbone.

Structures correspond 1:1 apart from the MLP: the checkpoint fuses gate and up
into `ffn.w12` (2*intermediate), which transformers splits into gate_proj and
up_proj. SwiGLU convention (x1, x2 = w12(x).chunk(2); silu(x1) * x2) puts gate
first.

Usage:
    python scripts/convert_sapiens2_to_hf.py \
        --src weights/sapiens2/model.safetensors --out weights/sapiens2_hf
"""
import argparse
import json
import os
import re
import shutil

import torch
from safetensors.torch import load_file, save_file


def convert(state):
    """Map original Sapiens2 keys onto transformers Sapiens2Backbone keys."""
    # Tolerate both a bare backbone checkpoint and a task checkpoint that
    # prefixes everything with "backbone." and carries a decode_head.
    st = {}
    for k, v in state.items():
        if k.startswith('decode_head.'):
            continue
        st[k[len('backbone.'):] if k.startswith('backbone.') else k] = v

    out, unmapped = {}, []
    for k, v in st.items():
        if k == 'cls_token':
            out['embeddings.cls_token'] = v
        elif k == 'storage_tokens':
            out['embeddings.register_tokens'] = v
        elif k.startswith('patch_embed.projection.'):
            out['embeddings.patch_embeddings.' + k.split('.')[-1]] = v
        elif k == 'ln1.weight':
            out['norm.weight'] = v
        elif k == 'rope_embed.periods':
            continue                      # transformers derives inv_freq from config
        elif k.startswith('blocks.'):
            m = re.match(r'blocks\.(\d+)\.(.+)', k)
            i, rest = m.group(1), m.group(2)
            p = f'model.layer.{i}.'
            if rest == 'ln1.weight':
                out[p + 'norm1.weight'] = v
            elif rest == 'ln2.weight':
                out[p + 'norm2.weight'] = v
            elif rest == 'attn.gamma.weight':
                out[p + 'layer_scale1.lambda1'] = v
            elif rest.startswith('attn.'):
                a = rest[len('attn.'):]
                name, _, suffix = a.rpartition('.')
                ren = {'wq': 'q_proj', 'wk': 'k_proj', 'wv': 'v_proj', 'proj': 'o_proj',
                       'q_norm': 'q_norm', 'k_norm': 'k_norm'}
                if name in ren:
                    out[p + f'attention.{ren[name]}.{suffix}'] = v
                else:
                    unmapped.append(k)
            elif rest.startswith('ffn.w12.'):
                suffix = rest.split('.')[-1]
                half = v.shape[0] // 2
                out[p + f'mlp.gate_proj.{suffix}'] = v[:half].clone()
                out[p + f'mlp.up_proj.{suffix}'] = v[half:].clone()
            elif rest.startswith('ffn.w3.'):
                out[p + f'mlp.down_proj.{rest.split(".")[-1]}'] = v
            else:
                unmapped.append(k)
        else:
            unmapped.append(k)
    return out, unmapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--config_dir', default=None,
                    help='Directory holding config.json (defaults to src dir)')
    args = ap.parse_args()

    cfg_dir = args.config_dir or os.path.dirname(args.src)
    os.makedirs(args.out, exist_ok=True)

    print(f'loading {args.src} ...', flush=True)
    state = load_file(args.src)
    converted, unmapped = convert(state)
    print(f'converted {len(converted)} tensors; {len(unmapped)} unmapped')
    if unmapped:
        print('  unmapped:', unmapped[:10])

    # Verify against the actual model definition before writing anything.
    from transformers import Sapiens2Backbone, Sapiens2Config
    cfg = Sapiens2Config.from_pretrained(cfg_dir)
    with torch.device('meta'):
        ref = Sapiens2Backbone(cfg)
    want = dict(ref.named_parameters())
    missing = [k for k in want if k not in converted]
    extra = [k for k in converted if k not in want]
    bad = [(k, tuple(want[k].shape), tuple(converted[k].shape))
           for k in want if k in converted and tuple(want[k].shape) != tuple(converted[k].shape)]
    print(f'missing={len(missing)} extra={len(extra)} shape_mismatch={len(bad)}')
    for lst, label in ((missing, 'MISSING'), (extra, 'EXTRA')):
        if lst:
            print(f'  {label}: {lst[:8]}')
    if bad:
        print('  SHAPE MISMATCH:', bad[:8])
    if missing or extra or bad:
        raise SystemExit('conversion incomplete -- refusing to write')

    for f in ('config.json', 'preprocessor_config.json'):
        s = os.path.join(cfg_dir, f)
        if os.path.exists(s):
            shutil.copy(s, os.path.join(args.out, f))
    # Backbone-only checkpoint: record the class so from_pretrained picks it up.
    cj = os.path.join(args.out, 'config.json')
    with open(cj) as f:
        c = json.load(f)
    c['architectures'] = ['Sapiens2Backbone']
    with open(cj, 'w') as f:
        json.dump(c, f, indent=2)

    dst = os.path.join(args.out, 'model.safetensors')
    save_file({k: v.contiguous() for k, v in converted.items()}, dst)
    print(f'wrote {dst}')


if __name__ == '__main__':
    main()
