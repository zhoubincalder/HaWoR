"""
Convert a `handpose-gold/1` bronze dataset into this repo's export layout.

One converter covers dexycb, ho3d and hot3d_rect because they share a protocol:
`clips.parquet` (per-clip camera + geometry), `frames.parquet` (per-frame camera
pose and a byte range into the shards), `annotations/<ver>/{train,valid}.parquet`
(per-frame MANO and keypoints) and `shards/shard-*.tar` (the JPEGs).

FOUR THINGS THE PROTOCOL DOES NOT SAY, EACH MEASURED HERE RATHER THAN ASSUMED
----------------------------------------------------------------------------

1.  `mano_pose[3:48]` is the residual for `flat_hand_mean=False`; this repo
    stores the FULL finger pose, because our MANOLayer is called with rotation
    matrices and so never adds `pose_mean` itself. The two differ by exactly
    `hands_mean`: measured as a constant 45-vector, std 0.000000 over frames,
    equal to -hands_mean to 2e-08. Feeding bronze poses in unchanged costs
    33.4mm (left) / 33.1mm (right); adding hands_mean brings it to 4e-06mm
    against this repo's own independently converted HOT3D labels.

2.  `mano_trans` is NaN in every row of every split we looked at, so global
    translation has to come from somewhere else. `kpts_3d` is fully populated,
    and the offset between it and our own MANO decode is a PURE translation --
    constant across all 21 joints to 3e-05mm. So translation is recovered as
    `mean(kpts_3d - joints(pose, betas, trans=0))`, which is exact, not a fit.

3.  `T_world_camera` is 7 floats with no stated order. Measured by reprojecting
    `kpts_3d` and comparing against the dataset's own `kpts_2d`: it is
    `[tx, ty, tz, qw, qx, qy, qz]` -- translation first, quaternion W FIRST.
    That ordering reprojects at 0.000px; the xyzw reading is off by 1e8 px and
    quaternion-first by 217px. `--verify` re-runs this check per clip.

4.  Principal points are NOT at the image centre: DexYCB is off by up to 17.6px
    and HO3D by 12.2px, with fx != fy by up to 1.34. The old export format only
    carried a scalar `focal.txt` and let the preprocess assume (W/2, H/2), which
    at 640x480 and f~600 is a ~15mm translation error at 0.5m -- larger than the
    model's own error. So this writes `intrinsics.txt` (fx fy cx cy), which
    hawor_preprocess_train.py prefers when present and falls back from when
    absent, leaving every previously converted dataset untouched.

WHAT IS WORTH TAKING
--------------------
Counting MANO-VALID frames rather than the `counts` field in dataset.json --
they are not the same number, and the gap is 160,696 frames:

    dexycb       463,160  (vs 29,116 converted here from a 4-of-10-subject
                           download, so this supersedes it outright)
    ho3d          83,325  (right hand only; not otherwise in this corpus)
    hot3d_rect   175,050  (LESS than the 227,400 already converted here: the
                           93,000-frame valid split has mano_valid=0 on BOTH
                           hands, so it is unusable for MANO supervision)

So hot3d_rect is deliberately NOT the default; the existing HOT3D export is
larger. It is still selectable, because it is rectified (fisheye624 -> pinhole,
upright) and that may matter more than frame count for some experiments.

Bronze `main` is written continuously by dagster and the annotations have
already moved v1.0.0 -> v2.0.0, so `--ref` pins a commit and defaults to doing
so rather than tracking a moving branch.
"""
import argparse
import io
import json
import os
import pickle
import shutil
import subprocess
import sys

sys.path.append(os.path.abspath('.'))

import joblib
import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq
import torch

REPO = 'lakefs://calder-dev'
BRONZE = 'bronze/program=third-party'

# load_gt_cam() right-multiplies the composed cam->world by this fixed +90deg
# rotation, which is HOT3D's raw Aria camera roll. Bronze clips are already
# stored upright, so we bake in the inverse and leave the shared loader alone.
R_90 = np.array([[0, 1, 0, 0],
                 [-1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]], dtype=np.float64)

# Left-hand shapedirs: smplx issue #48. A dataset whose betas were fitted with
# manopth's uncorrected sign needs the marker file so the preprocess matches.
# This is NOT hardcoded per project -- it is detected from the data by
# detect_left_shapedirs(), because getting it wrong is silent and costs ~1-4mm.
# For reference, what that detection returns today:
#     dexycb      manopth    (0.00004mm vs 1.26mm corrected)
#     hot3d_rect  corrected  (0.00003mm vs 3.84mm manopth)
#     ho3d        n/a        (right hand only; left is never valid)

MANO_LEFT_UNFIXED_MARKER = 'mano_left_unfixed'
NUM_JOINTS = 21

# Fingertips in OpenPose hand order. They are NOT MANO joints -- they are picked
# off the mesh, and datasets do not agree on which vertex. Measured on HO3D, the
# 16 non-tip joints reproduce our MANO decode exactly while tips 8/12/16/20 sit
# 1.9-3.9mm away, so averaging all 21 to recover translation drags it 0.63mm off
# on EVERY joint. Estimating from the non-tip set instead gives 0.0000mm.
TIP_JOINTS = (4, 8, 12, 16, 20)
TRANS_JOINTS = [i for i in range(NUM_JOINTS) if i not in TIP_JOINTS]


def lakectl(*args, check=True):
    return subprocess.run(['lakectl', *args], capture_output=True, text=True,
                          check=check)


def resolve_ref(ref):
    """Turn 'main' into the commit it currently points at, so a run is pinned."""
    if ref != 'main':
        return ref
    out = lakectl('log', f'{REPO}/main', '--amount', '1').stdout
    for line in out.splitlines():
        if line.startswith('ID:'):
            return line.split()[1]
    raise RuntimeError('could not resolve main to a commit id')


def fetch(ref, project, rel, dest):
    if os.path.exists(dest):
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + '.part'
    lakectl('fs', 'download', f'{REPO}/{ref}/{BRONZE}/project={project}/{rel}', tmp)
    os.replace(tmp, dest)
    return dest


def hands_mean(side):
    p = ('_DATA/data/mano/MANO_RIGHT.pkl' if side == 'r'
         else '_DATA/data_left/mano_left/MANO_LEFT.pkl')
    with open(p, 'rb') as f:
        return np.asarray(pickle.load(f, encoding='latin1')['hands_mean']).ravel()


_MANO_CACHE = {}


def mano_for(side, fix_shapedirs, device):
    key = (side, fix_shapedirs, device)
    if key in _MANO_CACHE:
        return _MANO_CACHE[key]
    from lib.models.mano_wrapper import MANO
    m = MANO(data_dir='_DATA/data/',
             model_path='_DATA/data/mano' if side == 'r' else '_DATA/data_left/mano_left',
             gender='neutral', num_hand_joints=15, create_body_pose=False,
             is_rhand=(side == 'r')).to(device)
    if side == 'l' and fix_shapedirs:
        m.shapedirs[:, 0, :] *= -1
    for p in m.parameters():
        p.requires_grad_(False)
    _MANO_CACHE[key] = m
    return m


def mano_joints(side, rot, pose_full, betas, fix_shapedirs, device, chunk=512):
    """(N,3),(N,45),(N,10) -> (N,21,3) at zero translation."""
    from hawor.utils.geometry import aa_to_rotmat
    m = mano_for(side, fix_shapedirs, device)
    out = []
    for i in range(0, rot.shape[0], chunk):
        r = torch.as_tensor(rot[i:i + chunk], dtype=torch.float32, device=device)
        p = torch.as_tensor(pose_full[i:i + chunk], dtype=torch.float32, device=device)
        b = torch.as_tensor(betas[i:i + chunk], dtype=torch.float32, device=device)
        n = r.shape[0]
        o = m(global_orient=aa_to_rotmat(r.reshape(-1, 3)).view(n, 1, 3, 3),
              hand_pose=aa_to_rotmat(p.reshape(-1, 3)).view(n, 15, 3, 3),
              betas=b, pose2rot=False)
        out.append(o.joints[:, :NUM_JOINTS].detach().cpu().numpy())
    return np.concatenate(out, 0)


def quat_wxyz_to_R(q):
    """(N,4) w-first quaternion -> (N,3,3). Order verified by reprojection."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)


def cam_to_world(T_world_camera):
    """(N,7) [t | q_wxyz] -> (N,4,4) camera->world."""
    T = np.asarray(T_world_camera, dtype=np.float64)
    M = np.zeros((T.shape[0], 4, 4))
    M[:, :3, :3] = quat_wxyz_to_R(T[:, 3:7])
    M[:, :3, 3] = T[:, 0:3]
    M[:, 3, 3] = 1.0
    return M


def detect_left_shapedirs(ann, device, n=200):
    """Decide the left-hand shapedirs convention by measuring both.

    Only the right convention makes `kpts_3d` a pure translation from our MANO
    decode; the wrong one leaves 1-4mm of shape error that no translation can
    absorb. The two options separate by 4-5 orders of magnitude, so this is a
    measurement rather than a heuristic. Returns True for corrected (smplx
    issue #48 applied), False for manopth's original sign, or None if the
    dataset has no valid left hand to measure.
    """
    mv = np.array(ann.column('mano_valid').to_pylist(), dtype=bool)
    idx = np.where(mv[:, 0])[0][:n]
    if len(idx) == 0:
        return None, None
    p = np.array(ann.column('mano_pose').take(idx).to_pylist(), dtype=np.float64)
    b = np.array(ann.column('mano_betas').take(idx).to_pylist(), dtype=np.float64)
    k3 = np.array(ann.column('kpts_3d').take(idx).to_pylist(), dtype=np.float64)
    hm = hands_mean('l')
    res = {}
    for fix in (True, False):
        J = mano_joints('l', p[:, 0, :3], p[:, 0, 3:] + hm, b[:, 0], fix, device)
        off = k3[:, 0] - J
        tr = off[:, TRANS_JOINTS].mean(1)
        res[fix] = float(np.linalg.norm(
            off[:, TRANS_JOINTS] - tr[:, None], axis=-1).mean())
    best = min(res, key=res.get)
    if res[best] > 1e-5 or res[not best] < res[best] * 100:
        raise RuntimeError(
            f'left shapedirs convention is ambiguous: corrected={res[True]*1000:.5f}mm '
            f'manopth={res[False]*1000:.5f}mm -- refusing to guess')
    return best, res


def load_tables(root, project, ann_version, splits):
    clips = pq.read_table(os.path.join(root, 'clips.parquet')).to_pylist()
    frames = pq.read_table(os.path.join(root, 'frames.parquet'))
    anns = []
    for s in splits:
        p = os.path.join(root, 'annotations', ann_version, f'{s}_annotations.parquet')
        if os.path.exists(p):
            anns.append(pq.read_table(p))
    if not anns:
        raise FileNotFoundError('no annotation parquet found for splits ' + str(splits))
    import pyarrow as pa
    return clips, frames, pa.concat_tables(anns, promote_options='default')


def build_index(table):
    """clip_id -> row indices, built once.

    Replaces a per-clip `Table.filter`, for two reasons. It was O(clips x rows)
    -- 7200 full scans of a 465,536-row table for DexYCB -- and, more bluntly,
    pyarrow 25.0.1 SEGFAULTS in Table.filter() on that table: 4 chunks of wide
    fixed_size_list columns. The crash takes the interpreter down with no
    traceback, and because Python buffers stdout when it is not a TTY, a run
    that dies this way leaves a log with no error in it at all. `take` on a
    precomputed index does not crash and is O(rows) once.
    """
    cid = table.column('clip_id').to_pylist()
    idx = {}
    for i, c in enumerate(cid):
        idx.setdefault(c, []).append(i)
    return {k: np.asarray(v, dtype=np.int64) for k, v in idx.items()}


def clip_rows(table, index, clip_id):
    rows = index.get(clip_id)
    if rows is None or len(rows) == 0:
        return table.slice(0, 0)
    import pyarrow as pa
    return table.take(pa.array(rows)).sort_by('frame_idx')


def extract_images(shard_path, rows, out_dir):
    """Pull JPEGs by byte range. frames.parquet gives (shard, offset, size)."""
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with open(shard_path, 'rb') as f:
        for idx, off, size in rows:
            f.seek(off)
            buf = f.read(size)
            if len(buf) != size:
                raise IOError(f'short read at {off} in {shard_path}')
            with open(os.path.join(out_dir, f'{idx:06d}.jpg'), 'wb') as g:
                g.write(buf)
            n += 1
    return n


def is_complete(out_dir):
    """True if a previous run finished this clip.

    convert_clip writes images, then anno.pth, then the small files with
    ego_extrinsics.pkl last, so that file is the completion marker. A clip
    interrupted mid-write lacks it and is redone.
    """
    return all(os.path.exists(os.path.join(out_dir, f))
               for f in ('anno.pth', 'ego_extrinsics.pkl', 'intrinsics.txt'))


def convert_clip(clip, ann, ann_idx, frames, fr_idx, out_root, project, device,
                 fix_shapedirs, shard_dir, verify):
    cid = clip['clip_id']
    name = clip.get('source_dir') or cid.replace('/', '_')
    out_dir = os.path.join(out_root, name)
    a = clip_rows(ann, ann_idx, cid)
    if a.num_rows == 0:
        return None, 'no annotations'
    fr = clip_rows(frames, fr_idx, cid)

    fidx = np.array(a.column('frame_idx').to_pylist())
    pose = np.array(a.column('mano_pose').to_pylist(), dtype=np.float64)    # (T,2,48)
    betas = np.array(a.column('mano_betas').to_pylist(), dtype=np.float64)  # (T,2,10)
    k3 = np.array(a.column('kpts_3d').to_pylist(), dtype=np.float64)        # (T,2,21,3)
    mvalid = np.array(a.column('mano_valid').to_pylist(), dtype=bool)       # (T,2)
    T = len(fidx)
    if not mvalid.any():
        return None, 'no valid MANO'

    anno = {}
    for side, slot in (('l', 0), ('r', 1)):
        rot = np.zeros((T, 3))
        pose45 = np.zeros((T, 45))
        bet = np.zeros((T, 10))
        trans = np.zeros((T, 3))
        m = mvalid[:, slot]
        if m.any():
            hm = hands_mean(side)
            rot[m] = pose[m, slot, :3]
            # (1) residual -> full finger pose
            pose45[m] = pose[m, slot, 3:] + hm
            bet[m] = betas[m, slot]
            # (2) translation recovered from kpts_3d; exact, not a fit -- but
            # only over the joints whose definition both sides share.
            J = mano_joints(side, rot[m], pose45[m], bet[m], fix_shapedirs, device)
            off = k3[m, slot] - J
            trans[m] = off[:, TRANS_JOINTS].mean(axis=1)
            resid = np.linalg.norm(
                off[:, TRANS_JOINTS] - trans[m][:, None], axis=-1).max()
            if resid > 1e-4:  # metres; every dataset so far lands at ~1e-8
                return None, (f'{side} hand: kpts_3d is not a pure translation '
                              f'from our MANO decode ({resid * 1000:.4f}mm)')
        # The preprocess reads validity as `any(rot != 0)`, so invalid frames
        # must leave rot exactly zero -- do not fill them with anything.
        anno[f'rot_{side}'] = torch.from_numpy(rot).float()
        anno[f'pose_{side}'] = torch.from_numpy(pose45).float()
        anno[f'betas_{side}'] = torch.from_numpy(bet).float()
        anno[f'trans_{side}'] = torch.from_numpy(trans).float()

    fx, fy, cx, cy = clip['intrinsics'][0]
    W, H = clip['width'], clip['height']

    # (3) camera->world, then bake in R_90^T so load_gt_cam's R_90 cancels.
    Tw = np.array(fr.column('T_world_camera').to_pylist(), dtype=np.float64)
    c2w = cam_to_world(Tw)[:T]
    if c2w.shape[0] < T:
        return None, f'frames.parquet has {c2w.shape[0]} poses for {T} annotated frames'
    stored = np.einsum('bij,jk->bik', c2w, R_90.T)

    if verify:
        err = reprojection_error(a, k3, mvalid, c2w, (fx, fy, cx, cy))
        if err is not None and err > verify:
            return None, f'reprojection {err:.3f}px > {verify}px'

    shard_ids = np.array(fr.column('shard').to_pylist())
    offs = np.array(fr.column('offset').to_pylist())
    sizes = np.array(fr.column('size').to_pylist())
    os.makedirs(out_dir, exist_ok=True)
    got = 0
    for sh in np.unique(shard_ids[:T]):
        sp = os.path.join(shard_dir, f'shard-{int(sh):05d}.tar')
        if not os.path.exists(sp):
            shutil.rmtree(out_dir, ignore_errors=True)
            return None, f'missing {os.path.basename(sp)}'
        sel = np.where(shard_ids[:T] == sh)[0]
        got += extract_images(sp, [(int(fidx[i]), int(offs[i]), int(sizes[i]))
                                   for i in sel],
                              os.path.join(out_dir, 'extracted_images'))
    if got != T:
        shutil.rmtree(out_dir, ignore_errors=True)
        return None, f'extracted {got} images for {T} frames'

    joblib.dump(anno, os.path.join(out_dir, 'anno.pth'))
    # focal.txt stays for readers that predate intrinsics.txt; the mean is what
    # the model's single-focal translation decode uses either way (fx and fy
    # differ by at most 0.22% in these datasets).
    with open(os.path.join(out_dir, 'focal.txt'), 'w') as f:
        f.write(str((fx + fy) / 2.0))
    # (4) true intrinsics, so the preprocess stops assuming a centred principal point
    with open(os.path.join(out_dir, 'intrinsics.txt'), 'w') as f:
        f.write(f'{fx} {fy} {cx} {cy}')
    with open(os.path.join(out_dir, 'head_pose.pkl'), 'wb') as f:
        pickle.dump(stored.astype(np.float32), f)
    with open(os.path.join(out_dir, 'ego_extrinsics.pkl'), 'wb') as f:
        pickle.dump(np.tile(np.eye(4, dtype=np.float32), (T, 1, 1)), f)
    if not fix_shapedirs:
        open(os.path.join(out_dir, MANO_LEFT_UNFIXED_MARKER), 'w').close()
    return name, T


def reprojection_error(a, k3, mvalid, c2w, K):
    """Project the dataset's own kpts_3d and compare to its own kpts_2d.

    Independent of anything this converter computes, so it catches a wrong
    quaternion order or a swapped principal point rather than confirming them.
    """
    if 'kpts_2d' not in a.schema.names:
        return None
    fx, fy, cx, cy = K
    k2 = np.array(a.column('kpts_2d').to_pylist(), dtype=np.float64)
    R = np.transpose(c2w[:, :3, :3], (0, 2, 1))
    t = c2w[:, :3, 3]
    errs = []
    for slot in (0, 1):
        m = mvalid[:, slot]
        if not m.any():
            continue
        Xc = np.einsum('tij,tnj->tni', R[m], k3[m, slot] - t[m][:, None])
        z = np.clip(Xc[..., 2], 1e-6, None)
        uv = np.stack([fx * Xc[..., 0] / z + cx, fy * Xc[..., 1] / z + cy], -1)
        ref = k2[m, slot, :, :2]
        ok = np.isfinite(ref).all(-1) & (Xc[..., 2] > 0)
        if ok.any():
            errs.append(np.abs(uv - ref)[ok].mean())
    return float(np.mean(errs)) if errs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', required=True,
                    choices=['dexycb', 'ho3d', 'hot3d_rect', 'hot3d', 'arctic'])
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--bronze_root', default=None,
                    help='local cache dir (default datasets/_bronze/<project>)')
    ap.add_argument('--ref', default='main',
                    help="commit id, or 'main' to resolve and pin it now")
    ap.add_argument('--ann_version', default='v2.0.0')
    ap.add_argument('--splits', nargs='+', default=['train', 'valid'])
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--verify', type=float, default=1.0,
                    help='max mean reprojection px per clip; 0 disables')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--keep_shards', action='store_true',
                    help='do not delete each shard after it is consumed')
    ap.add_argument('--no_resume', dest='resume', action='store_false',
                    help='re-convert clips that already look complete')
    args = ap.parse_args()

    ref = resolve_ref(args.ref)
    root = args.bronze_root or os.path.join('datasets', '_bronze', args.project)
    os.makedirs(root, exist_ok=True)
    print(f'bronze ref pinned at {ref}')

    for rel in ['dataset.json', 'clips.parquet', 'frames.parquet']:
        fetch(ref, args.project, rel, os.path.join(root, rel))
    for s in args.splits:
        fetch(ref, args.project, f'annotations/{args.ann_version}/{s}_annotations.parquet',
              os.path.join(root, 'annotations', args.ann_version, f'{s}_annotations.parquet'))

    clips, frames, ann = load_tables(root, args.project, args.ann_version, args.splits)
    keep = set(ann.column('clip_id').to_pylist())
    clips = [c for c in clips if c['clip_id'] in keep]
    if args.limit:
        clips = clips[:args.limit]
    print(f'{len(clips)} clips with annotations')

    fix_shapedirs, res = detect_left_shapedirs(ann, args.device)
    if fix_shapedirs is None:
        fix_shapedirs = True  # no left hand in this dataset; the flag is moot
        print('left shapedirs: no valid left hand, nothing to detect')
    else:
        print(f'left shapedirs: {"corrected" if fix_shapedirs else "manopth"} '
              f'(corrected {res[True] * 1000:.5f}mm vs manopth {res[False] * 1000:.5f}mm)')

    # Row indices, built once. See build_index() for why this is not a filter.
    ann_idx = build_index(ann)
    fr_idx = build_index(frames)

    shard_dir = os.path.join(root, 'shards')
    os.makedirs(shard_dir, exist_ok=True)
    os.makedirs(args.out_root, exist_ok=True)

    # Group clips by shard so each shard is fetched once, used, then dropped --
    # the full shard set is 46GB for dexycb alone.
    fr_shard = {}
    for cid, sh in zip(frames.column('clip_id').to_pylist(),
                       frames.column('shard').to_pylist()):
        fr_shard.setdefault(cid, set()).add(int(sh))
    order = sorted(clips, key=lambda c: min(fr_shard.get(c['clip_id'], {0})))

    done, skipped, manifest, resumed = 0, [], [], 0
    have = set()
    for c in order:
        # Resume: a completed clip is neither re-fetched nor re-converted, so an
        # interrupted run costs only the shards it had not reached.
        name = c.get('source_dir') or c['clip_id'].replace('/', '_')
        if args.resume and is_complete(os.path.join(args.out_root, name)):
            manifest.append(name)
            resumed += 1
            continue
        need = fr_shard.get(c['clip_id'], set())
        for sh in sorted(need):
            sp = os.path.join(shard_dir, f'shard-{sh:05d}.tar')
            if not os.path.exists(sp):
                if not args.keep_shards:
                    for old in sorted(have - need):
                        op = os.path.join(shard_dir, f'shard-{old:05d}.tar')
                        if os.path.exists(op):
                            os.remove(op)
                        have.discard(old)
                print(f'  fetching shard-{sh:05d}.tar')
                fetch(ref, args.project, f'shards/shard-{sh:05d}.tar', sp)
            have.add(sh)
        name, res = convert_clip(c, ann, ann_idx, frames, fr_idx,
                                 args.out_root, args.project,
                                 args.device, fix_shapedirs, shard_dir,
                                 args.verify or None)
        if name is None:
            skipped.append((c['clip_id'], res))
        else:
            manifest.append(name)
            done += 1
            if done % 50 == 0:
                print(f'  {done}/{len(order)} clips')

    with open(os.path.join(args.out_root, 'all.json'), 'w') as f:
        json.dump(sorted(manifest), f)
    with open(os.path.join(args.out_root, 'bronze_source.json'), 'w') as f:
        json.dump({'project': args.project, 'ref': ref,
                   'ann_version': args.ann_version, 'splits': args.splits,
                   'left_shapedirs': 'corrected' if fix_shapedirs else 'manopth',
                   'clips': len(manifest)}, f, indent=2)
    print(f'\nconverted {done} clips, reused {resumed} already complete, '
          f'skipped {len(skipped)}')
    for cid, why in skipped[:10]:
        print(f'  skip {cid}: {why}')
    if len(skipped) > 10:
        print(f'  ... and {len(skipped) - 10} more')


if __name__ == '__main__':
    main()
