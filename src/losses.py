import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Dice Loss for binary and multi-class segmentation tasks.
    Now supports 2D and 3D tensors.
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        """
        Args:
            preds (torch.Tensor): shape (N, C, ...) or (N, ...)
            targets (torch.Tensor): same shape or shape without channel (for CE)
        Returns:
            Dice loss (float)
        """
        # Ensure input is binary or multi-class logits
        if preds.dim() == 5:  # 3D segmentation (N, C, H, W, D)
            preds = torch.sigmoid(preds) if preds.shape[1] == 1 else F.softmax(preds, dim=1)
        elif preds.dim() == 4:  # 2D segmentation
            preds = torch.sigmoid(preds) if preds.shape[1] == 1 else F.softmax(preds, dim=1)
        else:
            raise ValueError("Predictions tensor must be 4D or 5D.")

        # If shape is (N, 1, ...) → squeeze channel
        if preds.shape[1] == 1:
            preds = preds.squeeze(1)
        if targets.dim() == preds.dim() + 1:
            targets = targets.squeeze(1)

        preds_flat = preds.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()

        dice_coeff = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_coeff

class CombinedLoss(nn.Module):
    """
    Combined Dice + Cross-Entropy Loss for binary segmentation in 3D or 2D.
    """
    def __init__(self, weight_dice=0.5, weight_ce=0.5, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.dice_loss = DiceLoss(smooth)
        self.ce_loss = nn.BCEWithLogitsLoss()  # binary case; if multi-class, switch to CrossEntropyLoss

    def forward(self, preds, targets):
        """
        Args:
            preds: shape (N, 1, H, W, D) or (N, 1, H, W) — raw logits
            targets: shape (N, 1, H, W, D) or (N, 1, H, W) — binary mask
        """
        # For BCEWithLogitsLoss we need shape: (N, 1, ...) and targets float
        ce_loss_value = self.ce_loss(preds, targets.float())

        # Dice uses sigmoid inside
        dice_loss_value = self.dice_loss(preds, targets)

        return self.weight_dice * dice_loss_value + self.weight_ce * ce_loss_value
