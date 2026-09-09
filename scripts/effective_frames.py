"""
Effective trainable frames per dataset -- what the windowing can actually reach.

"Usable frames" (a frame with at least one valid hand) overstates what training
sees, because HaworFullDataset does not sample frames. It finds maximal runs
where some hand is valid in EVERY frame, keeps only runs of at least seq_len,
and emits window starts every `stride` frames within them. Two consequences:

  * a valid frame sitting in a run shorter than 16 frames is unreachable. It is
    counted in "usable" and never trained on.
  * coverage depends on stride vs seq_len. Training uses CHUNK_STRIDE 8 with
    seq_len 16, so windows overlap and a kept run is fully covered -- and each
    frame is presented about twice per epoch. Validation uses VAL_CHUNK_STRIDE
    64, which is LARGER than the window, so val windows are disjoint samples
    with gaps: roughly a quarter of val frames are looked at.

Columns:
    frames        total frames in the fold's sequences
    usable        >= 1 valid hand in the frame
    effective     usable AND inside a kept run (>= seq_len) -- trainable
    stranded      usable - effective, i.e. valid but unreachable
    covered       frames inside at least one emitted window
    windows       windows emitted at this fold's stride
    presented     windows * seq_len, the frame-slots fed per epoch, counting
                  a frame once per window it appears in
    L / R         valid left / right hand instances among covered frames
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath('.'))

import numpy as np

DATASETS = [
    ('hot3d', 'datasets/hot3d_clips_export'),
    ('dexycb', 'datasets/dexycb_bronze_export'),
    ('arctic', 'datasets/arctic_export'),
    ('ho3d', 'datasets/ho3d_export'),
    ('h2o', 'datasets/h2o_export'),
    ('h2o3d', 'datasets/h2o3d_export'),
]
SEQ_LEN = 16


def runs_of_true(mask, min_len):
    """Maximal runs of True with length >= min_len, as (start, stop) exclusive."""
    out = []
    run = None
    n = len(mask)
    for t in range(n + 1):
        ok = t < n and mask[t]
        if ok and run is None:
            run = t
        elif not ok and run is not None:
            if t - run >= min_len:
                out.append((run, t))
            run = None
    return out


def window_starts(lo, hi, stride, seq_len):
    """Mirror HaworFullDataset._build_index for one kept run."""
    last = lo + (hi - lo) - seq_len
    st = list(range(lo, last + 1, stride))
    if (last - lo) % stride:
        st.append(last)
    return st


def fold_stats(root, seqs, stride):
    tot = usable = effective = covered = nwin = 0
    li = ri = 0
    for v in seqs:
        p = os.path.join(root, v, 'train_anno.npz')
        if not os.path.exists(p):
            continue
        with np.load(p) as d:
            valid = d['valid']              # (2, T)
        any_hand = valid.any(axis=0)
        T = len(any_hand)
        tot += T
        usable += int(any_hand.sum())
        cov = np.zeros(T, dtype=bool)
        for lo, hi in runs_of_true(any_hand, SEQ_LEN):
            effective += hi - lo
            for s in window_starts(lo, hi, stride, SEQ_LEN):
                cov[s:s + SEQ_LEN] = True
                nwin += 1
        covered += int(cov.sum())
        li += int(valid[0][cov].sum())
        ri += int(valid[1][cov].sum())
    return dict(frames=tot, usable=usable, effective=effective,
                stranded=usable - effective, covered=covered, windows=nwin,
                presented=nwin * SEQ_LEN, left=li, right=ri)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_stride', type=int, default=8)
    ap.add_argument('--val_stride', type=int, default=64)
    ap.add_argument('--json_out', default=None)
    args = ap.parse_args()

    hdr = (f'{"dataset":8} {"fold":5} {"frames":>9} {"usable":>9} {"effective":>10} '
           f'{"stranded":>9} {"covered":>9} {"windows":>8} {"presented":>10} '
           f'{"L":>8} {"R":>8}')
    print(hdr)
    print('-' * len(hdr))
    out, agg = {}, {}
    for name, root in DATASETS:
        if not os.path.isdir(root):
            continue
        out[name] = {}
        for fold in ('train', 'val', 'test'):
            f = os.path.join(root, f'{fold}.json')
            if not os.path.exists(f):
                continue
            stride = args.train_stride if fold == 'train' else args.val_stride
            s = fold_stats(root, json.load(open(f)), stride)
            s['stride'] = stride
            out[name][fold] = s
            print(f'{name:8} {fold:5} {s["frames"]:9} {s["usable"]:9} '
                  f'{s["effective"]:10} {s["stranded"]:9} {s["covered"]:9} '
                  f'{s["windows"]:8} {s["presented"]:10} {s["left"]:8} {s["right"]:8}')
            a = agg.setdefault(fold, {k: 0 for k in
                                      ('frames', 'usable', 'effective', 'stranded',
                                       'covered', 'windows', 'presented', 'left', 'right')})
            for k in a:
                a[k] += s[k]
    print('-' * len(hdr))
    for fold in ('train', 'val', 'test'):
        if fold not in agg:
            continue
        a = agg[fold]
        print(f'{"TOTAL":8} {fold:5} {a["frames"]:9} {a["usable"]:9} '
              f'{a["effective"]:10} {a["stranded"]:9} {a["covered"]:9} '
              f'{a["windows"]:8} {a["presented"]:10} {a["left"]:8} {a["right"]:8}')

    tr = agg.get('train', {})
    if tr:
        print(f'\ntrainable frames: {tr["effective"]} of {tr["usable"]} usable '
              f'({100 * tr["effective"] / max(tr["usable"], 1):.1f}%); '
              f'{tr["stranded"]} stranded in runs shorter than {SEQ_LEN}')
        print(f'hand instances trainable: {tr["left"] + tr["right"]} '
              f'({tr["left"]} left, {tr["right"]} right)')
        print(f'per epoch the loader feeds {tr["presented"]} frame-slots, '
              f'{tr["presented"] / max(tr["covered"], 1):.2f}x each covered frame')
    va = agg.get('val', {})
    if va:
        print(f'\nval at stride {args.val_stride} looks at {va["covered"]} of '
              f'{va["effective"]} effective frames '
              f'({100 * va["covered"] / max(va["effective"], 1):.1f}%) -- the stride '
              f'exceeds the {SEQ_LEN}-frame window, so val windows are disjoint samples')
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump({'per_dataset': out, 'totals': agg}, f, indent=2)
        print(f'\nwrote {args.json_out}')


if __name__ == '__main__':
    main()
