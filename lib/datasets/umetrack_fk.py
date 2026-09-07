"""
UmeTrack hand-rig forward kinematics, delegating to Meta's reference layer.

The rig SHOW3D and HOT3D share: 20 driven joints (5 fingers x 4 DoF), 17
skinning frames, 21 landmarks and a ~788-vertex mesh. Per-subject calibration
changes rest lengths and scale, not topology.

This is a thin wrapper over hot3d/data_loaders/umetrack_layer.py (Apache-2.0,
facebookresearch/hot3d) rather than a reimplementation, deliberately. A
hand-written version of this got three things wrong at once, none of them
guessable from the profile fields alone:

  * only joints 0..19 are driven; the profile ships 22 and the last two are not
    part of the finger chains
  * a joint's local transform ROTATES ABOUT its rest position
    (t = rest - R @ rest), it does not translate by it
  * the 17 frames are [wrist, wrist] followed by 3 per finger -- so
    `landmark_rest_bone_indices` addresses frames, not joints, and frame 1 is
    the wrist base, driven by no joint

Getting any of those wrong put the landmarks 126-303mm out. Since the profile
is a serialisation of the reference dataclass, the reference is the source of
truth for how to consume it.

Handedness: the profile is a LEFT-hand model. The right hand is obtained by
negating column 0 of the wrist transform (`wrist[:, 0] *= -1`), per
UmeTrackHandDataProvider. Passing an identity wrist for both hands leaves the
right hand 204mm out.

Verified against SHOW3D's own `landmarks_3d_mm_local`, which is worth doing
rather than assuming: v1 of that dataset shipped landmarks undersized by a
per-subject scale factor. Measured 0.00000mm for both hands.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), 'hot3d'))

from data_loaders.umetrack_layer import get_skinning_weights, skin_points  # noqa: E402

N_FRAMES = 17


class UmeTrackRig:
    """Poseable UmeTrack hand rig for one subject."""

    def __init__(self, hand_model: dict):
        t = lambda k, dt=torch.float64: torch.tensor(hand_model[k], dtype=dt)  # noqa: E731
        self.axes = t('joint_rotation_axes')                 # (22,3)
        self.rest = t('joint_rest_positions')                # (22,3)
        self.lm_rest = t('landmark_rest_positions')          # (21,3)
        self.lm_w = t('landmark_rest_bone_weights')          # (21,3)
        self.lm_i = t('landmark_rest_bone_indices', torch.int64)
        self.verts = t('mesh_vertices')                      # (V,3)
        self.tris = t('mesh_triangles', torch.int64)
        self.dense_w = t('dense_bone_weights')               # (V,17)
        self.limits = t('joint_limits')                      # (22,2)
        self.scale = float(np.asarray(hand_model['hand_scale']))

        # Landmarks carry sparse (index, weight) pairs; the layer wants a dense
        # (1, 21, 17) matrix over frames.
        self.lm_skin = get_skinning_weights(
            self.lm_i[None], self.lm_w[None], N_FRAMES)

    def _skin(self, joint_angles, points, skin_mat, wrist=None):
        a = torch.as_tensor(joint_angles, dtype=torch.float64).reshape(1, -1)
        w = (torch.eye(4, dtype=torch.float64)[None] if wrist is None
             else torch.as_tensor(wrist, dtype=torch.float64).reshape(1, 4, 4))
        out = skin_points(
            joint_rest_positions=self.rest[None],
            joint_rotation_axes=self.axes[None],
            skin_mat=skin_mat,
            joint_angles=a,
            points=points[None],
            wrist_transforms=w,
        )
        return out[0].numpy()

    @staticmethod
    def wrist_for(side, wrist=None):
        """Wrist transform with handedness applied.

        The profile models a LEFT hand; a right hand is the same model with
        column 0 of the wrist transform negated.
        """
        W = np.eye(4) if wrist is None else np.asarray(wrist, dtype=np.float64).copy()
        if side in ('r', 'right', 1):
            W[:, 0] *= -1
        return W

    def landmarks(self, joint_angles, side='left', wrist=None):
        """(22,) angles -> (21,3) landmarks, mm. wrist=None gives wrist-local."""
        return self._skin(joint_angles, self.lm_rest, self.lm_skin,
                          self.wrist_for(side, wrist))

    def mesh(self, joint_angles, side='left', wrist=None):
        """(22,) angles -> (V,3) skinned vertices, mm."""
        return self._skin(joint_angles, self.verts, self.dense_w[None],
                          self.wrist_for(side, wrist))


def load_rig(path):
    with open(path) as f:
        return UmeTrackRig(json.load(f)['hand_model'])
