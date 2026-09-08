"""
Re-cut H2O's train/val folds to be SUBJECT-disjoint. Leaves test alone.

H2O's published split (`label_split/pose_{train,val,test}.txt`, which
h2o_to_export.py follows verbatim) puts subject3 on BOTH sides of the
train/val line: 24 sequences / 14,235 usable frames in train, another 24 /
14,676 in val. So its val number answers "a new recording of a hand already
fitted" rather than "a hand never seen" -- a subject's shape is a fixed set of
MANO betas the model learns directly, so this measures an easier task than the
one Stage-A faces at inference.

The test fold is untouched and stays subject4, which appears nowhere else.

WHAT THIS COSTS. H2O has only three non-test subjects, so any whole-subject
holdout is about a third of the non-test data:

    option                          train     val   val%   discarded
    val = subject3 whole  (chosen)  54,694  28,911  34.6%          0
    val = subject3's official half  54,694  14,676  21.2%     14,235
    val = subject2 whole            56,765  26,840  32.1%          0
    published split (contaminated)  68,929  14,676     --          0

subject3 whole is chosen because it discards nothing and moves the fewest
sequences: subject1 and subject2 are already wholly in train. A 34.6% val fold
is larger than convention, but those frames are used, not wasted -- subsample
at eval time if it costs too much wall-clock, rather than deleting data here.

Training loses 14,235 usable frames (-20.7% of H2O, ~1.6% of the whole corpus).

DELIBERATE CONSEQUENCE: val numbers from this split are not comparable to
published H2O results, which use the contaminated fold. `split_by_subject.json`
records that, and the original manifests are kept as `*_official.json` so the
published protocol can be reproduced.
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

VAL_SUBJECTS = ('subject3',)


def subject_of(seq_name):
    return seq_name.split('__')[0]


def usable_frames(export_root, seq):
    npz = os.path.join(export_root, seq, 'train_anno.npz')
    if not os.path.exists(npz):
        return None
    d = np.load(npz)
    return int(d['valid'].any(0).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export_root', default='datasets/h2o_export')
    ap.add_argument('--val_subjects', nargs='+', default=list(VAL_SUBJECTS))
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    root = args.export_root
    # Pool the existing train+val manifests and re-cut them. test is read only
    # to prove it stays disjoint; it is never rewritten.
    pool, test = [], []
    for name, dest in (('train', pool), ('val', pool), ('test', test)):
        # Prefer the untouched originals if this has already been run once, so
        # re-running is idempotent rather than compounding.
        for cand in (f'{name}_official.json', f'{name}.json'):
            p = os.path.join(root, cand)
            if os.path.exists(p):
                dest.extend(json.load(open(p)))
                break

    ready = [s for s in sorted(set(pool)) if usable_frames(root, s) is not None]
    dropped = sorted(set(pool) - set(ready))
    if dropped:
        print(f'WARNING: {len(dropped)} sequences have no train_anno.npz, excluded')

    val_s = set(args.val_subjects)
    missing = val_s - {subject_of(s) for s in ready}
    if missing:
        sys.exit(f'val subjects not present: {sorted(missing)}')

    val = [s for s in ready if subject_of(s) in val_s]
    train = [s for s in ready if subject_of(s) not in val_s]

    stats = {}
    for k, ids in (('train', train), ('val', val), ('test', test)):
        f = sum(usable_frames(root, s) or 0 for s in ids)
        subs = sorted({subject_of(s) for s in ids})
        stats[k] = (len(ids), f, subs)
        print(f'{k:6} {len(ids):4} seq  {f:7} usable  subjects {subs}')

    tr, va, te = (set(stats[k][2]) for k in ('train', 'val', 'test'))
    for a, b, an, bn in ((tr, va, 'train', 'val'), (tr, te, 'train', 'test'),
                         (va, te, 'val', 'test')):
        ov = sorted(a & b)
        print(f'{an} n {bn}: {ov if ov else "DISJOINT"}')
        if ov:
            sys.exit(f'refusing to write: {an}/{bn} share {ov}')

    if not args.write:
        print('\n(dry run -- pass --write)')
        return

    # Preserve the published manifests once, so the official protocol remains
    # reproducible after this rewrite.
    for name in ('train', 'val'):
        src = os.path.join(root, f'{name}.json')
        keep = os.path.join(root, f'{name}_official.json')
        if os.path.exists(src) and not os.path.exists(keep):
            shutil.copy2(src, keep)

    for name, ids in (('train', train), ('val', val)):
        with open(os.path.join(root, f'{name}.json'), 'w') as f:
            json.dump(sorted(ids), f)
    with open(os.path.join(root, 'split_by_subject.json'), 'w') as f:
        json.dump({
            'scheme': 'subject-disjoint (re-cut from H2O official split)',
            'val_subjects': sorted(val_s),
            'train_subjects': stats['train'][2],
            'test_subjects': stats['test'][2],
            'n_train_seq': stats['train'][0], 'n_val_seq': stats['val'][0],
            'usable_train_frames': stats['train'][1],
            'usable_val_frames': stats['val'][1],
            'supersedes': 'H2O label_split/pose_{train,val}.txt '
                          '(kept as train_official.json / val_official.json)',
            'note': 'val is NOT comparable to published H2O numbers: the '
                    'official fold shares subject3 between train and val.',
        }, f, indent=2)
    print('\nwrote train.json, val.json, split_by_subject.json '
          '(originals kept as *_official.json)')


if __name__ == '__main__':
    main()
