"""
Split the HOT3D clip export into train/val folds that are SUBJECT-disjoint.

The previous split was cut by source recording (`split_by_sequence.json`, 890/110).
That keeps a recording whole on one side, which stops adjacent frames of the same
take from straddling the fold -- but it does not stop the same *person* appearing
on both sides. Measured on that split, all five val participants (P0001, P0003,
P0009, P0010, P0011) also occur in train. So the val number answered "a new
recording of a hand you have already fitted", not "a hand you have never seen".
Since a subject's hand shape is a fixed set of MANO betas that the model gets to
learn directly, that is an optimistic measurement of exactly the thing Stage-A
has to do at inference.

Expanding the corpus from 1000 to 1516 clips brought in three participants that
were absent or nearly absent before (P0012, P0014, P0015), which makes a
subject-disjoint fold possible for the first time. P0015 is held out whole:
152 clips, ~10% of the corpus, matching the 11% the old val fold had.

Consequence worth stating plainly: val numbers from this split are NOT comparable
to any run measured on `split_by_sequence.json`, and should be expected to look
worse. That is the point -- the previous number was measuring an easier task.

Source recording comes from `000000.info.json` inside each clip tar, field
`sequence_id`, formatted `P####_<hash>`; the participant is the part before the
underscore.
"""
import argparse
import json
import os
import subprocess
import sys

VAL_PARTICIPANTS = ('P0015',)


def sequence_id(tar_path):
    """Read `sequence_id` out of a clip tar without unpacking the whole thing."""
    # --occurrence=1 lets tar stop at the first match instead of scanning all
    # ~150 frames' worth of members.
    raw = subprocess.run(
        ['tar', 'xOf', tar_path, '--occurrence=1', '000000.info.json'],
        capture_output=True, check=True).stdout
    return json.loads(raw)['sequence_id']


def frame_count(export_root, clip):
    import numpy as np
    npz = os.path.join(export_root, clip, 'train_anno.npz')
    if not os.path.exists(npz):
        return 0
    return int(np.load(npz)['n_frames'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--clips_dir', default='datasets/hot3d_clips/train_aria')
    ap.add_argument('--export_root', default='datasets/hot3d_clips_export')
    ap.add_argument('--val_participants', nargs='+', default=list(VAL_PARTICIPANTS))
    ap.add_argument('--write', action='store_true',
                    help='without this, report the split but change nothing')
    args = ap.parse_args()

    clips = sorted(f[:-4] for f in os.listdir(args.clips_dir) if f.endswith('.tar'))
    if not clips:
        sys.exit(f'no clip tars under {args.clips_dir}')

    seq = {c: sequence_id(os.path.join(args.clips_dir, c + '.tar')) for c in clips}
    part = {c: seq[c].split('_')[0] for c in clips}

    val_p = set(args.val_participants)
    unknown = val_p - set(part.values())
    if unknown:
        sys.exit(f'val participants not present in the corpus: {sorted(unknown)}')

    # Only keep clips that actually converted; a clip without train_anno.npz
    # would silently become a missing sample at load time.
    ready = [c for c in clips
             if os.path.exists(os.path.join(args.export_root, c, 'train_anno.npz'))]
    missing = sorted(set(clips) - set(ready))
    if missing:
        print(f'WARNING: {len(missing)} clips have no train_anno.npz, excluded: '
              f'{missing[:5]}{"..." if len(missing) > 5 else ""}')

    val = [c for c in ready if part[c] in val_p]
    train = [c for c in ready if part[c] not in val_p]

    tr_f = sum(frame_count(args.export_root, c) for c in train)
    va_f = sum(frame_count(args.export_root, c) for c in val)

    tr_p = sorted({part[c] for c in train})
    va_p = sorted({part[c] for c in val})
    overlap = sorted(set(tr_p) & set(va_p))

    print(f'train {len(train):5d} clips  {tr_f:7d} frames  participants {tr_p}')
    print(f'val   {len(val):5d} clips  {va_f:7d} frames  participants {va_p}')
    print(f'val fraction: {len(val) / len(ready):.1%} of clips, '
          f'{va_f / (tr_f + va_f):.1%} of frames')
    print(f'participant overlap: {overlap if overlap else "NONE (subject-disjoint)"}')
    if overlap:
        sys.exit('refusing to write: folds share participants')

    if not args.write:
        print('\n(dry run -- pass --write to update train.json / val.json)')
        return

    for name, ids in (('train', train), ('val', val)):
        with open(os.path.join(args.export_root, f'{name}.json'), 'w') as f:
            json.dump(ids, f)
    meta = {
        'scheme': 'subject-disjoint',
        'val_participants': sorted(val_p),
        'train_participants': tr_p,
        'n_train_clips': len(train), 'n_val_clips': len(val),
        'n_train_frames': tr_f, 'n_val_frames': va_f,
        'supersedes': 'split_by_sequence.json',
        'clip_to_sequence': {c: seq[c] for c in ready},
    }
    with open(os.path.join(args.export_root, 'split_by_subject.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print('\nwrote train.json, val.json, split_by_subject.json')


if __name__ == '__main__':
    main()
