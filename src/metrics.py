import torch


def dice_coefficient(preds, targets, smooth=1e-5):
    """
    Compute the Dice Coefficient.

    Args:
        preds (torch.Tensor): Predicted tensor of shape (N, C, H, W, D).
        targets (torch.Tensor): Ground truth tensor of the same shape as preds.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: Dice Coefficient score.
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum()

    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()


def iou_score(preds, targets, smooth=1e-5):
    """
    Compute the Intersection over Union (IoU) score.

    Args:
        preds (torch.Tensor): Predicted tensor of shape (N, C, H, W, D).
        targets (torch.Tensor): Ground truth tensor of the same shape as preds.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        float: IoU score.
    """
    preds = torch.sigmoid(preds)
    preds = (preds > 0.5).float()

    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()
