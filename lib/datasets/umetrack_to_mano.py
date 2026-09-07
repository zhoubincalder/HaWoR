"""
Fit MANO to a posed UmeTrack hand rig.

SHOW3D annotates hands with the UmeTrack rig, not MANO, so every frame has to
be re-expressed before HaWoR can train on it. The two are not the same model --
UmeTrack has 20 single-axis joints with limits, MANO has 15 free 3-DoF
rotations plus a PCA shape space -- so this is a fit with a residual, not a
change of variables. The point of this module is to measure that residual
honestly.

Structure follows the data: shape is a property of the SUBJECT (SHOW3D ships 38
calibrated profiles), pose is per frame. So betas are fitted once per subject
and then frozen, and only the 48 pose parameters vary per frame. That is 38
shape fits rather than 3.5M joint shape+pose optimisations.

Correspondence is 20 of 21 landmarks. UmeTrack's `palm_center` has no MANO
counterpart and MANO's `thumb_mcp` has no UmeTrack one, so both are dropped --
UmeTrack models the thumb with one fewer joint than MANO.

Fitting targets landmarks rather than the mesh. The rig does expose a
788-vertex skinned surface (umetrack_fk.UmeTrackRig.mesh), which would
constrain the fit better, but the two meshes have no vertex correspondence, so
using it needs a closest-point term.

MEASURED (subject KHA829, 32-64 frames per hand, both hands):

    overall mean          ~5.4 mm
    fingertips            0.9 - 1.8 mm
    PIP / DIP             2 - 4 mm
    MCP                   6 - 12 mm
    wrist                 15 mm

So finger articulation transfers well and the palm does not: the two rigs place
the wrist and MCP joints differently, which no amount of pose fitting resolves.
Running 4x the iterations made the overall figure WORSE (5.46 -> 6.55 mm) --
the optimiser is already trading wrist error against finger error, not failing
to converge, so this is a floor of the representation rather than of the solver.

Two consequences worth stating plainly:

  * these labels are approximate. Every other converter in this directory
    reproduces its dataset's own ground truth to ~1e-4 mm; this one cannot,
    because MANO simply is not that rig.
  * 5.4mm is the same order as the models being trained (best crop PA-MPJPE
    4.31mm, full-frame LoRA 6.73mm). Data converted this way is usable for
    pretraining and robustness, but its label noise is comparable to the signal,
    so it cannot be used to measure accuracy or to push it.
"""
import numpy as np
import torch

# UmeTrack landmark index -> MANO/OpenPose joint index.
# UmeTrack: 0..4 tips (thumb,index,middle,ring,pinky), 5 wrist,
#           6..7 thumb, 8..10 index, 11..13 middle, 14..16 ring, 17..19 pinky,
#           20 palm centre.
# MANO (after HaWoR's mano_to_openpose remap): 0 wrist, then thumb/index/
#           middle/ring/little as mcp,pip,dip,tip.
UT_TO_MANO = [
    (5, 0),                                    # wrist
    (8, 5), (9, 6), (10, 7), (1, 8),           # index
    (11, 9), (12, 10), (13, 11), (2, 12),      # middle
    (14, 13), (15, 14), (16, 15), (3, 16),     # ring
    (17, 17), (18, 18), (19, 19), (4, 20),     # pinky / little
    (6, 2), (7, 3), (0, 4),                    # thumb (no counterpart for mcp)
]
UT_IDX = np.array([u for u, _ in UT_TO_MANO])
MANO_IDX = np.array([m for _, m in UT_TO_MANO])


def build_mano(side, device='cpu'):
    """HaWoR's MANO wrapper, instantiated once (run_mano rebuilds per call)."""
    import sys, os
    sys.path.append(os.path.abspath('.'))
    from lib.models.mano_wrapper import MANO
    cfg = {
        'data_dir': '_DATA/data/',
        'model_path': '_DATA/data/mano' if side == 'right' else '_DATA/data_left/mano_left',
        'gender': 'neutral',
        'num_hand_joints': 15,
        'create_body_pose': False,
        'is_rhand': side == 'right',
    }
    m = MANO(**cfg).to(device)
    if side == 'left':
        # smplx issue #48, as run_mano_left does. SHOW3D has no MANO betas of
        # its own, so we are free to fit in the corrected convention -- which
        # is what the rest of the pipeline uses, so no unfixed marker is needed.
        m.shapedirs[:, 0, :] *= -1
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _joints(mano, betas, rot, pose, transl):
    # smplx's MANO here takes rotation matrices, not axis-angle (pose2rot=True
    # is not honoured), which is also how the rest of this codebase calls it.
    from hawor.utils.geometry import aa_to_rotmat
    N = rot.shape[0]
    out = mano(global_orient=aa_to_rotmat(rot.reshape(-1, 3)).view(N, 1, 3, 3),
               hand_pose=aa_to_rotmat(pose.reshape(-1, 3)).view(N, 15, 3, 3),
               betas=betas, transl=transl, pose2rot=False)
    return out.joints[:, :21]


def fit(rig, joint_angles, side, device='cpu', iters=(400, 250),
        lr=(0.05, 0.05), seed=0, betas=None):
    """Fit MANO to a batch of UmeTrack poses for ONE subject and hand.

    joint_angles: (N, 22) array of per-frame rig angles.
    betas:        if given, held fixed (per-subject shape already solved).

    Returns (betas (10,), rot (N,3), pose (N,45), transl (N,3), err_mm (N,)).
    err_mm is the mean per-landmark distance over the 20 corresponded points.
    """
    torch.manual_seed(seed)
    ja = np.asarray(joint_angles, dtype=np.float64)
    N = ja.shape[0]

    # Target: rig landmarks in wrist-local frame, mm -> m.
    tgt = np.stack([rig.landmarks(a, side) for a in ja])[:, UT_IDX] / 1000.0
    tgt = torch.tensor(tgt, dtype=torch.float32, device=device)

    mano = build_mano(side, device)
    fixed_betas = betas is not None
    b = (torch.tensor(betas, dtype=torch.float32, device=device)[None].clone()
         if fixed_betas else torch.zeros(1, 10, device=device))
    b.requires_grad_(not fixed_betas)
    rot = torch.zeros(N, 3, device=device, requires_grad=True)
    pose = torch.zeros(N, 45, device=device, requires_grad=True)
    tr = torch.zeros(N, 3, device=device, requires_grad=True)

    # Stage 1: wrist only -- gets global orientation and translation into the
    # right basin before the fingers start moving. Fitting all of it at once
    # lets the fingers absorb a global misalignment.
    opt = torch.optim.Adam([rot, tr] + ([] if fixed_betas else [b]), lr=lr[0])
    for _ in range(iters[0]):
        opt.zero_grad()
        j = _joints(mano, b.expand(N, -1), rot, pose, tr)[:, MANO_IDX]
        loss = ((j - tgt) ** 2).sum(-1).mean()
        loss.backward()
        opt.step()

    # Stage 2: everything, with a light pose prior to keep the fingers from
    # contorting into a lower-residual but implausible configuration.
    params = [rot, pose, tr] + ([] if fixed_betas else [b])
    opt = torch.optim.Adam(params, lr=lr[1])
    for _ in range(iters[1]):
        opt.zero_grad()
        j = _joints(mano, b.expand(N, -1), rot, pose, tr)[:, MANO_IDX]
        loss = ((j - tgt) ** 2).sum(-1).mean() + 1e-4 * (pose ** 2).mean()
        if not fixed_betas:
            loss = loss + 1e-3 * (b ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        j = _joints(mano, b.expand(N, -1), rot, pose, tr)[:, MANO_IDX]
        err = torch.norm(j - tgt, dim=-1).mean(-1) * 1000.0
    return (b[0].detach().cpu().numpy(), rot.detach().cpu().numpy(),
            pose.detach().cpu().numpy(), tr.detach().cpu().numpy(),
            err.cpu().numpy())
