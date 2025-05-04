import torch
import torch.nn as nn


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error loss computed only over valid regions of a mask.

    Args:
        ignore_indices (list[int]): List of values in the mask to ignore during loss computation.
    """
    def __init__(self, ignore_indices):
        super(MaskedMSELoss, self).__init__()
        self.ignore_indices = ignore_indices

    def forward(self, s2_pred, s2_true, s2_mask):
        """
        Args:
            s2_pred (Tensor): Predicted S2 values, shape (N, 1, H, W) or (N, H, W)
            s2_true (Tensor): Ground truth S2 values, shape (N, H, W)
            s2_mask (Tensor): Mask tensor with values to ignore, shape (N, H, W)

        Returns:
            Tensor: Scalar masked MSE loss
        """
        # Create mask where all ignore_indices are excluded
        valid_mask = torch.ones_like(s2_mask, dtype=torch.bool)
        for idx in self.ignore_indices:
            valid_mask &= (s2_mask != idx)

        # Ensure prediction has correct shape
        if s2_pred.ndim == 4:
            s2_pred = s2_pred.squeeze(1)

        s2_pred_valid = s2_pred[valid_mask]
        s2_true_valid = s2_true[valid_mask]

        loss = torch.mean((s2_pred_valid - s2_true_valid) ** 2)
        return loss
