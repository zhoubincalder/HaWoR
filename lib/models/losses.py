"""
Losses for the camera-space hand motion estimator.

These follow the HaMeR/HMR2 convention: every loss reduces with `sum()` over the
whole batch (not `mean()`), so the magnitude of each term scales with B*T. The
LOSS_WEIGHTS in the config are calibrated for that convention -- if you change
the reduction here you must re-tune them.
"""
import torch
import torch.nn as nn


def _make_loss_fn(loss_type: str) -> nn.Module:
    if loss_type == 'l1':
        return nn.L1Loss(reduction='none')
    elif loss_type == 'l2':
        return nn.MSELoss(reduction='none')
    else:
        raise NotImplementedError(f'Unsupported loss function: {loss_type}')


class Keypoint2DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        2D keypoint loss in normalized crop coordinates, i.e. both prediction and
        ground truth live in [-0.5, 0.5] (see HAWOR.forward_step).
        """
        super().__init__()
        self.loss_fn = _make_loss_fn(loss_type)

    def forward(self, pred_keypoints_2d: torch.Tensor, gt_keypoints_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_keypoints_2d (torch.Tensor): (B, J, 2) predicted 2D keypoints.
            gt_keypoints_2d (torch.Tensor): (B, J, 3) ground truth keypoints, last
                channel is the per-joint confidence.
        Returns:
            torch.Tensor: scalar loss, summed over batch and joints.
        """
        conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.loss_fn(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).sum(dim=(1, 2))
        return loss.sum()


class Keypoint3DLoss(nn.Module):

    def __init__(self, loss_type: str = 'l1'):
        """
        Root-relative 3D keypoint loss. Both prediction and ground truth are
        aligned on joint `pelvis_id` (the wrist, joint 0, for MANO) before the
        comparison, so this term never supervises the global translation.
        """
        super().__init__()
        self.loss_fn = _make_loss_fn(loss_type)

    def forward(self, pred_keypoints_3d: torch.Tensor, gt_keypoints_3d: torch.Tensor,
                pelvis_id: int = 0) -> torch.Tensor:
        """
        Args:
            pred_keypoints_3d (torch.Tensor): (B, J, 3) predicted 3D keypoints.
            gt_keypoints_3d (torch.Tensor): (B, J, 4) ground truth keypoints, last
                channel is the per-joint confidence.
        Returns:
            torch.Tensor: scalar loss, summed over batch and joints.
        """
        gt_keypoints_3d = gt_keypoints_3d.clone()
        pred_keypoints_3d = pred_keypoints_3d - pred_keypoints_3d[:, pelvis_id, :].unsqueeze(1)
        gt_keypoints_3d[:, :, :-1] = gt_keypoints_3d[:, :, :-1] - gt_keypoints_3d[:, pelvis_id, :-1].unsqueeze(1)
        conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
        loss = (conf * self.loss_fn(pred_keypoints_3d, gt_keypoints_3d[:, :, :-1])).sum(dim=(1, 2))
        return loss.sum()


class ParameterLoss(nn.Module):

    def __init__(self):
        """
        MSE loss on MANO parameters (rotation matrices and betas).
        """
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction='none')

    def forward(self, pred_param: torch.Tensor, gt_param: torch.Tensor,
                has_param: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            pred_param (torch.Tensor): (B, D) predicted parameters.
            gt_param (torch.Tensor): (B, D) ground truth parameters.
            has_param (torch.Tensor): (B,) 0/1 mask marking samples that actually
                carry a ground truth annotation. Defaults to all ones.
        Returns:
            torch.Tensor: scalar loss, summed over batch and dimensions.
        """
        batch_size = pred_param.shape[0]
        num_dims = len(pred_param.shape)
        mask_dimension = [batch_size] + [1] * (num_dims - 1)
        if has_param is None:
            has_param = pred_param.new_ones(batch_size)
        has_param = has_param.type(pred_param.type()).view(*mask_dimension)
        loss_param = has_param * self.loss_fn(pred_param, gt_param)
        return loss_param.sum()
