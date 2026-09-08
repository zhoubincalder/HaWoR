"""
Write CORPUS_MANIFEST.json: what each export actually contains, measured.

This exists because an archive was trusted on its sha256 alone. The tarball on
LakeFS matched its local twin byte for byte -- and both were a snapshot taken
before 516 HOT3D clips were added, so the check proved integrity of the wrong
content and 78,000 frames had to be re-downloaded. A checksum answers "did this
transfer intact"; it cannot answer "is this the corpus I think it is".

So the manifest records per-dataset sequence and frame counts, fold membership
and subject lists, and the git commit that produced them. archive_corpus.sh
verifies each archive against these counts before marking it done, and a
consumer can check a restore the same way.
"""
import json
import os
import subprocess
import sys

import numpy as np

# The superseded 4-of-10-subject DexYCB download is deliberately absent: it is
# kept locally only as a convention reference for the left-shapedirs check, and
# shipping it beside dexycb_bronze_export would invite training on both, which
# would duplicate subjects 01/02/03/06.
DATASETS = [
    ('hot3d', 'hot3d_clips_export'),
    ('dexycb', 'dexycb_bronze_export'),
    ('arctic', 'arctic_export'),
    ('ho3d', 'ho3d_export'),
    ('h2o', 'h2o_export'),
    ('h2o3d', 'h2o3d_export'),
]


def git_sha():
    try:
        return subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def subject_of(name):
    if '__' in name:
        return name.split('__')[0]
    import re
    m = re.match(r'[A-Za-z]+', name)
    return m.group(0) if m else name


def describe(root):
    """Measured contents of one export tree."""
    if not os.path.isdir(root):
        return None
    seqs = sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))
    folds = {}
    for split in ('train', 'val', 'test'):
        f = os.path.join(root, f'{split}.json')
        if not os.path.exists(f):
            continue
        ids = json.load(open(f))
        usable = inst = frames = 0
        for c in ids:
            z = os.path.join(root, c, 'train_anno.npz')
            if not os.path.exists(z):
                continue
            d = np.load(z)
            v = d['valid']
            frames += int(d['n_frames'])
            usable += int(v.any(0).sum())
            inst += int(v.sum())
        folds[split] = {
            'sequences': len(ids), 'frames': frames,
            'usable_frames': usable, 'hand_instances': inst,
            'subjects': sorted({subject_of(c) for c in ids}),
        }
    src = os.path.join(root, 'bronze_source.json')
    entry = {
        'sequences_on_disk': len(seqs),
        'preprocessed': sum(
            os.path.exists(os.path.join(root, s, 'train_anno.npz')) for s in seqs),
        'first_sequence': seqs[0] if seqs else None,
        'last_sequence': seqs[-1] if seqs else None,
        'folds': folds,
        'mano_left_unfixed': any(
            os.path.exists(os.path.join(root, s, 'mano_left_unfixed'))
            for s in seqs[:50]),
    }
    if os.path.exists(src):
        entry['bronze_source'] = json.load(open(src))
    return entry


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else 'datasets/_archives/CORPUS_MANIFEST.json'
    man = {'git_commit': git_sha(), 'datasets': {}}
    tot = {'train': 0, 'val': 0, 'test': 0}
    for name, d in DATASETS:
        e = describe(os.path.join('datasets', d))
        if e is None:
            print(f'[skip] {d} absent')
            continue
        e['export_dir'] = d
        man['datasets'][name] = e
        for s in tot:
            tot[s] += e['folds'].get(s, {}).get('usable_frames', 0)
        parts = '  '.join(
            f'{s} {e["folds"][s]["usable_frames"]}' for s in ('train', 'val', 'test')
            if s in e['folds'])
        print(f'{name:8} {e["sequences_on_disk"]:5} seq  {parts}')
    man['totals_usable_frames'] = tot
    print(f'\ntotal usable: train {tot["train"]}  val {tot["val"]}  test {tot["test"]}')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(man, f, indent=2)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
