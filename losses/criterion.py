# ------------------------------------------------------------------------------
# https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/master/lib/models/losses/loss.py
# and https://github.com/XuJiacong/PIDNet
# ------------------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.nn import functional as F
from configs import config


class CrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss for models with multiple outputs, supporting balancing weights.
    """
    def __init__(self, ignore_index=-1, weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def _compute_loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def forward(self, predictions, target):
        if config.MODEL.NUM_OUTPUTS == 1:
            predictions = [predictions]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        aux_weights = config.LOSS.AUX_WEIGHTS

        if len(balance_weights) == len(predictions):
            return sum(w * self._compute_loss(pred, target) for w, pred in zip(balance_weights, predictions))
        elif len(predictions) == 1:
            return aux_weights * self._compute_loss(predictions[0], target)
        else:
            raise ValueError("Mismatch between number of predictions and balance weights.")


class OHEMCrossEntropyLoss(nn.Module):
    """
    Online Hard Example Mining (OHEM) Cross-Entropy Loss for semantic segmentation.
    """
    def __init__(self, ignore_index=-1, threshold=0.7, min_kept=100000, weight=None):
        super().__init__()
        self.threshold = threshold
        self.min_kept = max(1, min_kept)
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction='none')

    def _standard_loss(self, prediction, target):
        return self.loss_fn(prediction, target)

    def _ohem_loss(self, prediction, target):
        prob = F.softmax(prediction, dim=1)
        pixel_losses = self.loss_fn(prediction, target).view(-1)
        valid_mask = target.view(-1) != self.ignore_index

        temp_target = target.clone()
        temp_target[temp_target == self.ignore_index] = 0
        prob = prob.gather(1, temp_target.unsqueeze(1))
        prob, indices = prob.view(-1)[valid_mask].sort()
        min_value = prob[min(self.min_kept, prob.numel() - 1)]
        thresh = max(min_value, self.threshold)

        pixel_losses = pixel_losses[valid_mask][indices]
        pixel_losses = pixel_losses[prob < thresh]
        return pixel_losses.mean()

    def forward(self, predictions, target):
        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        balance_weights = config.LOSS.BALANCE_WEIGHTS
        aux_weights = config.LOSS.AUX_WEIGHTS

        if len(balance_weights) == len(predictions):
            loss_fns = [self._standard_loss] * (len(balance_weights) - 1) + [self._ohem_loss]
            return sum(w * fn(pred, target) for w, pred, fn in zip(balance_weights, predictions, loss_fns))
        elif len(predictions) == 1:
            return aux_weights * self._ohem_loss(predictions[0], target)
        else:
            raise ValueError("Mismatch between number of predictions and balance weights.")


def weighted_bce(prediction, target):
    """
    Weighted binary cross-entropy for boundary prediction.
    """
    n, c, h, w = prediction.size()
    logit = prediction.permute(0, 2, 3, 1).contiguous().view(1, -1)
    target_flat = target.view(1, -1)

    pos_mask = (target_flat == 1)
    neg_mask = (target_flat == 0)

    weight = torch.zeros_like(logit)
    pos_count = pos_mask.sum()
    neg_count = neg_mask.sum()
    total = pos_count + neg_count
    weight[pos_mask] = neg_count.float() / total
    weight[neg_mask] = pos_count.float() / total

    return F.binary_cross_entropy_with_logits(logit, target_flat, weight, reduction='mean')




class BoundaryDiceLoss(nn.Module):
    """
    Combines weighted BCE and Dice loss for boundary prediction.
    """
    def __init__(self, bce_weight=20.0, smooth=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth

    def dice_loss(self, prediction, target):
        pred = torch.sigmoid(prediction).contiguous().view(-1)
        tgt = target.contiguous().view(-1)
        intersection = (pred * tgt).sum()
        dice_score = (2. * intersection + self.smooth) / (pred.sum() + tgt.sum() + self.smooth)
        return 1 - dice_score

    def forward(self, prediction, target):
        bce = self.bce_weight * weighted_bce(prediction, target)
        dice = self.dice_loss(prediction, target)
        return bce + dice


        
        
        


