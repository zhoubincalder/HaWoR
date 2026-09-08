"""
Emit train.json / val.json for a converted bronze export, using bronze's split.

bronze_to_export.py merges the annotation splits it was asked for into one
`all.json`, because the split is a property of the annotation version rather
than of the conversion. This reattaches it.

Bronze's own train/valid folds are already SUBJECT-disjoint, which is checked
here rather than trusted -- it is the property that makes a val number mean
"a hand you have never seen", and it is the one the previous HOT3D split turned
out not to have. Measured:

    dexycb   6400 train / 800 valid clips, valid = 20201015-subject-09 alone
    ho3d       49 train /   6 valid clips, valid = MC and ND

So no re-splitting is needed or wanted; re-cutting these would only risk
breaking a property they already hold.
"""
import argparse
import json
import os
import sys

import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export_root', required=True)
    ap.add_argument('--bronze_root', required=True)
    ap.add_argument('--ann_version', default='v2.0.0')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    clips = pq.read_table(os.path.join(args.bronze_root, 'clips.parquet'),
                          columns=['clip_id', 'source_dir', 'subject']).to_pylist()
    # The export directory is named after source_dir when bronze provides one.
    name_of = {c['clip_id']: (c.get('source_dir') or c['clip_id'].replace('/', '_'))
               for c in clips}
    subj_of = {c['clip_id']: c['subject'] for c in clips}

    converted = set(json.load(open(os.path.join(args.export_root, 'all.json'))))
    folds, subjects = {}, {}
    for split, out in (('train', 'train'), ('valid', 'val')):
        p = os.path.join(args.bronze_root, 'annotations', args.ann_version,
                         f'{split}_annotations.parquet')
        if not os.path.exists(p):
            continue
        ids = set(pq.read_table(p, columns=['clip_id']).column('clip_id').to_pylist())
        # Only clips that actually converted; a name in the manifest that has no
        # train_anno.npz would become a missing sample at load time.
        folds[out] = sorted(name_of[c] for c in ids
                            if c in name_of and name_of[c] in converted)
        subjects[out] = {subj_of[c] for c in ids if c in subj_of}

    if 'train' not in folds or 'val' not in folds:
        sys.exit(f'need both splits, got {sorted(folds)}')

    overlap = subjects['train'] & subjects['val']
    dup = set(folds['train']) & set(folds['val'])
    for k in ('train', 'val'):
        print(f'{k:6} {len(folds[k]):6} clips  {len(subjects[k])} subjects')
    print(f'val subjects: {sorted(subjects["val"])}')
    print(f'subject overlap: {sorted(overlap) if overlap else "NONE (subject-disjoint)"}')
    if overlap:
        sys.exit('refusing to write: folds share subjects')
    if dup:
        sys.exit(f'refusing to write: {len(dup)} clips in both folds')

    missing = converted - set(folds['train']) - set(folds['val'])
    if missing:
        print(f'WARNING: {len(missing)} converted clips in neither fold')

    if not args.write:
        print('\n(dry run -- pass --write)')
        return
    for k in ('train', 'val'):
        with open(os.path.join(args.export_root, f'{k}.json'), 'w') as f:
            json.dump(folds[k], f)
    with open(os.path.join(args.export_root, 'split_by_subject.json'), 'w') as f:
        json.dump({'scheme': 'bronze annotation split (verified subject-disjoint)',
                   'ann_version': args.ann_version,
                   'val_subjects': sorted(subjects['val']),
                   'train_subjects': sorted(subjects['train']),
                   'n_train_clips': len(folds['train']),
                   'n_val_clips': len(folds['val'])}, f, indent=2)
    print('\nwrote train.json, val.json, split_by_subject.json')


if __name__ == '__main__':
    main()
