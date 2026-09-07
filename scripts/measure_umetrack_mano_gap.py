"""
Measure how well MANO can represent a UmeTrack hand, three ways.

This is the evidence behind the decision not to build a SHOW3D training path.
SHOW3D annotates hands with the UmeTrack rig; HaWoR needs MANO. The question is
what that conversion costs, and the answer is ~5mm no matter how it is done --
because MANO's 10-dimensional shape space cannot represent an individually
calibrated UmeTrack hand more closely than ~4mm of surface distance.

HOT3D is the measuring stick: it publishes BOTH umetrack_pose and mano_pose for
the same hand on the same frame, so the disagreement can be measured directly
rather than inferred from a fit. The repo's own test samples carry two
participants on two devices (Aria and Quest3), which is enough to separate
subject-specific effects from rig-definition ones.

Three measurements, in increasing order of how hard they try:

  direct    Pose the rig, run MANO with HOT3D's published parameters, and
            compare the 20 corresponded landmarks. No fitting. Answers "how far
            apart are the two conventions".

  offset    Express that disagreement in the wrist frame and ask whether it is
            a fixed vector. It is, to 0.06-0.45mm within a subject -- but the
            vector DIFFERS between subjects, so a universal correction table
            does not work (cross-calibration is as bad as no calibration). This
            is the measurement that kills the cheap fix.

  surface   Fit MANO's surface to the rig's 788-vertex mesh by bidirectional
            Chamfer, which sidesteps joint naming entirely, and score the
            resulting joints against HOT3D's MANO. This is the most favourable
            method available and still lands at ~5mm.

Results as of the run that informed the decision:

  direct    left 5.94mm, right 6.68mm (Aria); 5.79 / 6.68 (Quest3)
            worst: wrist 13-17mm, thumb_distal 9-14mm, middle_prox 10-12mm
            best:  thumb_intermed / middle_intermed 1.2-3.5mm
            only 1 of 20 landmarks agrees under 5mm across all four
            subject/hand combinations (ring_tip). Scoring on one subject alone
            flatters it to 5 of 20 -- which is why both samples matter, and why
            a per-joint mask does not rescue this.

  offset    within-subject SD 0.06-0.45mm  (systematic)
            cross-subject residual 5.5-6.5mm  (not shared)

  surface   surface residual 3.7-4.0mm; joints vs HOT3D MANO 4.8-5.2mm
            (one of four fits diverged from a zero init -- initialise from a
            landmark fit if this is ever used in earnest)

The 3.7-4.0mm surface figure is the cleanest number here, because it compares
against exact geometry rather than against HOT3D's own MANO fit, which carries
its own residual. Note that HOT3D's two annotation tracks disagree by 5.94 /
6.68mm: Meta had both representations and every reason to make them agree, and
did not do better. ~5mm is the state of the art for this conversion, not a
deficiency of this code.

Bearing on HaWoR: the best crop model scores 4.31mm PA-MPJPE, so labels
converted this way are noisier than the model they would supervise. Usable for
pretraining -- label noise does not block representation learning, and SHOW3D is
the only in-the-wild egocentric data available -- but not for measuring or
improving accuracy.

Usage:
    python scripts/measure_umetrack_mano_gap.py            # all three
    python scripts/measure_umetrack_mano_gap.py --only direct
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.abspath('.'))

# HOT3D's own UmeTrack-landmark -> smplx-MANO-joint map (mano_layer.py:31).
# 20 of 21; UmeTrack's palm_center has no MANO counterpart.
UT_TO_MANO = [16, 17, 18, 19, 20, 0, 14, 15, 1, 2, 3, 4, 5, 6, 10, 11, 12, 7, 8, 9]
NAMES = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip', 'wrist',
         'thumb_intermed', 'thumb_distal', 'index_prox', 'index_intermed',
         'index_distal', 'middle_prox', 'middle_intermed', 'middle_distal',
         'ring_prox', 'ring_intermed', 'ring_distal', 'pinky_prox',
         'pinky_intermed', 'pinky_distal']

SAMPLES = {
    'Aria P0003': 'Aria/P0003_c701bd11',
    'Quest3 P0002': 'Quest3/P0002_273c2819',
}
BASE = ('https://raw.githubusercontent.com/facebookresearch/hot3d/main/'
        'hot3d/data_loaders/tests/data_sample')
FILES = ['mano_hand_pose_trajectory.jsonl', 'umetrack_hand_pose_trajectory.jsonl',
         'umetrack_hand_user_profile.json']


def fetch(cache):
    """HOT3D's paired test samples: both annotation tracks plus the rig profile."""
    import urllib.request
    for label, rel in SAMPLES.items():
        d = os.path.join(cache, rel.replace('/', '_'))
        os.makedirs(d, exist_ok=True)
        for f in FILES:
            p = os.path.join(d, f)
            if not os.path.exists(p):
                urllib.request.urlretrieve(f'{BASE}/{rel}/{f}', p)
        yield label, d


def q2R(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mano_layer(side, pca, device='cpu'):
    """smplx MANO as HOT3D builds it, including the smplx #48 left fix."""
    import smplx
    path = ('_DATA/data/mano/MANO_RIGHT.pkl' if side == 'right'
            else '_DATA/data_left/mano_left/MANO_LEFT.pkl')
    kw = dict(use_pca=True, num_pca_comps=15) if pca else dict(use_pca=False)
    m = smplx.create(path, 'mano', is_rhand=(side == 'right'), **kw).to(device)
    if side == 'left':
        ref = smplx.create('_DATA/data/mano/MANO_RIGHT.pkl', 'mano', is_rhand=True,
                           **kw)
        if torch.sum(torch.abs(m.shapedirs[:, 0, :].cpu() - ref.shapedirs[:, 0, :])) < 1:
            m.shapedirs[:, 0, :] *= -1
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_pair(root):
    from lib.datasets.umetrack_fk import UmeTrackRig
    with open(os.path.join(root, 'umetrack_hand_user_profile.json')) as f:
        rig = UmeTrackRig(json.load(f)['hand_model'])
    tracks = {}
    for key, fn in (('ut', 'umetrack_hand_pose_trajectory.jsonl'),
                    ('mn', 'mano_hand_pose_trajectory.jsonl')):
        tracks[key] = {}
        with open(os.path.join(root, fn)) as f:
            for line in f:
                d = json.loads(line)
                tracks[key][d['timestamp_ns']] = d['hand_poses']
    ts = sorted(set(tracks['ut']) & set(tracks['mn']))
    return rig, tracks['ut'], tracks['mn'], ts


def rig_world(rig, u, side, what='landmarks'):
    """Rig landmarks or mesh in world coordinates, metres.

    The profile is a LEFT-hand model; the right hand mirrors x. The wrist
    transform is in metres while the rig is in millimetres, so the mirror and
    the unit change are applied to the local points before the wrist transform.
    """
    p = (rig.landmarks(u['joint_angles'], 'left') if what == 'landmarks'
         else rig.mesh(u['joint_angles'], 'left')) / 1000.0
    if side == 'right':
        p = p * np.array([-1.0, 1.0, 1.0])
    R, t = q2R(u['wrist_xform']['q_wxyz']), np.asarray(u['wrist_xform']['t_xyz'])
    return (R @ p.T).T + t


def mano_joints_21(layer, m, device='cpu'):
    """HOT3D's published MANO for one frame -> 21 joints (16 regressed + 5 tips)."""
    import cv2
    from smplx.vertex_ids import vertex_ids
    tips = list(vertex_ids['mano'].values())
    aa = cv2.Rodrigues(q2R(m['wrist_xform']['q_wxyz']))[0].reshape(3)
    T = lambda x: torch.tensor(x, dtype=torch.float32, device=device)[None]  # noqa: E731
    o = layer(betas=T(m['betas']), global_orient=T(aa), hand_pose=T(m['pose']),
              transl=T(m['wrist_xform']['t_xyz']), return_verts=True)
    return torch.cat([o.joints[0], o.vertices[0][tips]], 0).detach().cpu().numpy()


def per_frame_offsets(rig, ut, mn, ts):
    """(MANO joint - UmeTrack landmark) per frame, rotated into the wrist frame."""
    layers = {s: mano_layer(s, pca=True) for s in ('left', 'right')}
    out = {'left': [], 'right': []}
    for t in ts:
        for hi, side in (('0', 'left'), ('1', 'right')):
            u, m = ut[t].get(hi), mn[t].get(hi)
            if not u or not m or u.get('hand_confidence', 0) < 0.5:
                continue
            utw = rig_world(rig, u, side)[:20]
            mj = mano_joints_21(layers[side], m)[UT_TO_MANO]
            R = q2R(u['wrist_xform']['q_wxyz'])
            out[side].append((R.T @ (mj - utw).T).T)      # into the wrist frame
    return {k: (np.stack(v) if v else None) for k, v in out.items()}


def report_direct(data):
    print('\n=== direct: the two conventions, no fitting ===')
    tables = {}
    for label, off in data.items():
        for side in ('left', 'right'):
            if off[side] is None:
                continue
            mag = np.linalg.norm(off[side], axis=-1) * 1000
            tables[(label, side)] = mag.mean(0)
            print(f'  {side:<6} {label:<14} n={len(off[side]):>3}  '
                  f'mean {mag.mean():6.2f} mm  median {np.median(mag.mean(-1)):6.2f}')
    if tables:
        print(f"\n  {'landmark':<17}" + ''.join(f'{l[:4]}/{s[:1]:<5}'
                                                for l, s in tables))
        under = 0
        for i, n in enumerate(NAMES):
            vals = [t[i] for t in tables.values()]
            ok = max(vals) < 5.0
            under += ok
            print(f'  {n:<17}' + ''.join(f'{v:8.1f}' for v in vals)
                  + ('   ok' if ok else ''))
        print(f'\n  landmarks under 5mm everywhere: {under} of 20')


def report_offset(data):
    print('\n=== offset: is the disagreement a fixed vector? ===')
    print(f"  {'':<24}{'raw':>9}{'self-cal':>10}{'cross-cal':>11}")
    labels = list(data)
    for side in ('left', 'right'):
        for i, label in enumerate(labels):
            O = data[label][side]
            other = data[labels[1 - i]][side] if len(labels) > 1 else None
            if O is None or other is None:
                continue
            f = lambda a: np.linalg.norm(a, axis=-1).mean() * 1000  # noqa: E731
            print(f'  {side:<6} {label:<17}{f(O):7.2f}mm'
                  f'{f(O - O.mean(0, keepdims=True)):9.2f}'
                  f'{f(O - other.mean(0, keepdims=True)):10.2f}')
    print('  self-cal is a lower bound (fitted and scored on the same frames).')
    print('  cross-cal is the number that matters: it is no better than raw, so')
    print('  the offset is subject-specific and a shared correction table fails.')


def report_surface(cache, device):
    print('\n=== surface: MANO fitted to the rig mesh, scored on HOT3D MANO ===')
    for label, root in fetch(cache):
        rig, ut, mn, ts = load_pair(root)
        for hi, side in (('0', 'left'), ('1', 'right')):
            frames = [t for t in ts if ut[t].get(hi) and mn[t].get(hi)
                      and ut[t][hi].get('hand_confidence', 0) >= 0.5][:24]
            if not frames:
                continue
            tgt = torch.tensor(np.stack([rig_world(rig, ut[t][hi], side, 'mesh')
                                         for t in frames]),
                               dtype=torch.float32, device=device)
            ref = mano_layer(side, pca=True, device=device)
            gt = np.stack([mano_joints_21(ref, mn[t][hi], device) for t in frames])
            L = mano_layer(side, pca=False, device=device)
            from smplx.vertex_ids import vertex_ids
            tips = list(vertex_ids['mano'].values())
            N = len(frames)
            b = torch.zeros(1, 10, device=device, requires_grad=True)
            rot = torch.zeros(N, 3, device=device, requires_grad=True)
            pose = torch.zeros(N, 45, device=device, requires_grad=True)
            tr = torch.tensor(
                np.stack([ut[t][hi]['wrist_xform']['t_xyz'] for t in frames]),
                dtype=torch.float32, device=device).clone().requires_grad_(True)

            def fwd():
                o = L(betas=b.expand(N, -1), global_orient=rot, hand_pose=pose,
                      transl=tr, return_verts=True)
                return o.vertices, torch.cat([o.joints, o.vertices[:, tips]], 1)

            def cham(a, c):
                d = torch.cdist(a, c)
                return d.min(2).values.mean() + d.min(1).values.mean()

            for stage, (ps, it, lr) in enumerate((([rot, tr, b], 300, 0.05),
                                                  ([rot, pose, tr, b], 400, 0.02))):
                opt = torch.optim.Adam(ps, lr=lr)
                for _ in range(it):
                    opt.zero_grad()
                    V, _ = fwd()
                    loss = cham(V, tgt)
                    if stage:
                        loss = loss + 1e-4 * (pose ** 2).mean() + 1e-3 * (b ** 2).mean()
                    loss.backward()
                    opt.step()
            with torch.no_grad():
                V, J = fwd()
                surf = cham(V, tgt).item() * 1000 / 2
                j = J[:, :21].cpu().numpy()
            absj = np.linalg.norm(j - gt, axis=-1).mean() * 1000
            rel = np.linalg.norm((j - j[:, :1]) - (gt - gt[:, :1]), axis=-1).mean() * 1000
            flag = '   <- diverged, needs a landmark-fit init' if absj > 20 else ''
            print(f'  {label:<14} {side:<6} surface {surf:5.2f} mm | '
                  f'joints abs {absj:6.2f} root-rel {rel:6.2f}{flag}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--only', choices=['direct', 'offset', 'surface'], default=None)
    p.add_argument('--cache', default='/tmp/hot3d_paired')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()

    if a.only in (None, 'direct', 'offset'):
        data = {}
        for label, root in fetch(a.cache):
            rig, ut, mn, ts = load_pair(root)
            data[label] = per_frame_offsets(rig, ut, mn, ts)
        if a.only in (None, 'direct'):
            report_direct(data)
        if a.only in (None, 'offset'):
            report_offset(data)
    if a.only in (None, 'surface'):
        report_surface(a.cache, a.device)


if __name__ == '__main__':
    main()
