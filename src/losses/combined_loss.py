import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss (Lin et al., 2017).

    Class weights compensate for class imbalance, while the focal term
    down-weights easy examples and focuses learning on hard pixels.
    """

    def __init__(self, class_weights=None, gamma=2.0, ignore_index=-1):
        super().__init__()
        self.class_weights = class_weights
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()

        valid_mask = targets != self.ignore_index

        safe_targets = targets.clone()
        safe_targets[~valid_mask] = 0

        ce_loss = F.nll_loss(
            log_probs,
            safe_targets,
            weight=self.class_weights,
            reduction="none",
        )

        pt = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)

        focal_term = (1.0 - pt).pow(self.gamma)
        loss = focal_term * ce_loss

        loss = loss[valid_mask]

        if loss.numel() == 0:
            return logits.sum() * 0.0

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined segmentation loss.

        loss = alpha * FocalLoss + (1 - alpha) * DiceLoss

    Focal Loss:
        - handles class imbalance
        - focuses on difficult pixels

    Dice Loss:
        - optimizes region overlap
        - computed only over foreground classes
    """

    def __init__(
        self,
        weight=None,
        ignore_index=-1,
        alpha=0.5,
        gamma=2.0,
    ):
        super().__init__()

        self.focal_loss = FocalLoss(
            class_weights=weight,
            gamma=gamma,
            ignore_index=ignore_index,
        )

        self.ignore_index = ignore_index
        self.alpha = alpha
        self.smooth = 1e-5

    def forward(self, logits, targets):

        focal = self.focal_loss(logits, targets)

        probs = F.softmax(logits, dim=1)
        num_classes = logits.shape[1]

        valid_mask = targets != self.ignore_index

        dice_loss = 0.0
        valid_classes = 0

        # Skip background (class 0)
        for c in range(1, num_classes):

            pred_c = probs[:, c][valid_mask]
            target_c = (targets == c)[valid_mask].float()

            union = pred_c.sum() + target_c.sum()

            if union > 0:
                intersection = (pred_c * target_c).sum()

                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

                dice_loss += 1.0 - dice
                valid_classes += 1

        if valid_classes > 0:
            dice_loss /= valid_classes
        else:
            dice_loss = logits.new_tensor(0.0)

        return self.alpha * focal + (1.0 - self.alpha) * dice_loss