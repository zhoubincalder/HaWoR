"""
Write true per-sequence intrinsics for ARCTIC, whose export assumed a centred
principal point.

ARCTIC's egocentric camera has cx=421.23, cy=268.89 on an 840x600 frame. The
converter stored only focal.txt, so hawor_preprocess_train.py fell back to
(W/2, H/2) = (420, 300) -- cy wrong by 31.11 px, 5% of image height. That
shifts j2d, j2d_conf and the GT-derived boxes across all 193,134 ARCTIC frames,
and gives the full-frame model a wrong img_center for its translation decode.

MEASURED, against bronze's own kpts_2d on s05__box_grab_01, non-tip joints:

    our j2d as-is                 left 28.11 px   right 31.14 px
    our j2d + true principal pt   left  9.42 px   right  0.03 px

The right hand collapsing to 0.03 px is what confirms the diagnosis. Note the
raw comparison looked like 70-75 px until a 10-frame indexing offset between
our export and bronze's clips was accounted for; without that alignment the
error looks far worse than it is and the cause is invisible.

The left hand keeps a 9.42 px residual after correction (~6.5 mm at this focal
and depth). That is a SEPARATE, unexplained issue -- possibly the left-hand
shapedirs convention, since arctic_export carries no mano_left_unfixed marker
-- and this script does not address it.

WHY intrinsics.txt RATHER THAN WARPING THE IMAGES. h2o and h2o3d solve the same
problem in their converters by translating every frame so the principal point
lands at the centre (cv2.warpAffine). That is equally correct, but redoing it
for ARCTIC would resample 193k images and lose a little sharpness for no
benefit, now that the preprocess reads intrinsics.txt when present. The two
datasets end up handling it differently, which is worth knowing when reading
the exports: h2o/h2o3d images are shifted, ARCTIC's intrinsics are declared.

Source of truth is bronze's arctic clips.parquet, which carries per-clip
intrinsics for exactly the 301 sequences in this export.
"""
import argparse
import json
import os
import re
import subprocess
import sys

REPO = 'lakefs://calder-dev'
CLIPS = ('bronze/program=third-party/project=arctic/clips.parquet')


def export_name(source_dir):
    """'arctic__s01-box_grab_01__0-23733' -> 's01__box_grab_01'."""
    m = re.match(r'arctic__(s\d+)-(.+?)__\d+-\d+$', source_dir)
    if not m:
        return None
    return f'{m.group(1)}__{m.group(2)}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export_root', default='datasets/arctic_export')
    ap.add_argument('--clips', default=None,
                    help='local arctic clips.parquet (downloaded if absent)')
    ap.add_argument('--ref', default='main')
    ap.add_argument('--write', action='store_true')
    args = ap.parse_args()

    import pyarrow.parquet as pq

    path = args.clips or 'datasets/_bronze/arctic/clips.parquet'
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f'fetching {CLIPS}')
        subprocess.run(['lakectl', 'fs', 'download',
                        f'{REPO}/{args.ref}/{CLIPS}', path], check=True)

    rows = pq.read_table(path, columns=['clip_id', 'source_dir', 'intrinsics',
                                        'width', 'height']).to_pylist()
    by_name = {}
    for r in rows:
        n = export_name(r['source_dir'])
        if n:
            by_name[n] = r

    seqs = sorted(d for d in os.listdir(args.export_root)
                  if os.path.isdir(os.path.join(args.export_root, d)))
    missing = [s for s in seqs if s not in by_name]
    if missing:
        print(f'WARNING: no bronze intrinsics for {len(missing)} sequences: '
              f'{missing[:4]}')

    worst = (0.0, None)
    written = 0
    for s in seqs:
        r = by_name.get(s)
        if r is None:
            continue
        fx, fy, cx, cy = r['intrinsics'][0]
        W, H = r['width'], r['height']
        # Sanity: the export's own recorded size must agree, or these
        # intrinsics describe a different image than the one on disk.
        import numpy as np
        z = os.path.join(args.export_root, s, 'train_anno.npz')
        if os.path.exists(z):
            with np.load(z) as d:
                eW, eH = (int(x) for x in d['img_size'])
            if (eW, eH) != (W, H):
                print(f'SKIP {s}: export is {eW}x{eH}, bronze says {W}x{H}')
                continue
        d = max(abs(cx - W / 2), abs(cy - H / 2))
        if d > worst[0]:
            worst = (d, s)
        if args.write:
            with open(os.path.join(args.export_root, s, 'intrinsics.txt'), 'w') as f:
                f.write(f'{fx} {fy} {cx} {cy}')
            written += 1

    print(f'{len(seqs)} sequences, {len(by_name)} matched in bronze')
    print(f'largest principal-point offset from centre: {worst[0]:.2f} px ({worst[1]})')
    if not args.write:
        print('\n(dry run -- pass --write, then re-run the preprocess with --overwrite)')
        return
    print(f'wrote intrinsics.txt for {written} sequences')
    print('\nnow re-run:\n  uv run python lib/datasets/hawor_preprocess_train.py '
          f'--video_root {args.export_root} --set_file train.json --overwrite\n'
          '  (and again for val.json)')


if __name__ == '__main__':
    main()
