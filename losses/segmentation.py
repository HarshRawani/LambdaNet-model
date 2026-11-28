import math

import torch
import torch.nn as nn


class OHEMCrossEntropy(nn.Module):
    """Online hard-example mining: average the loss over only the hardest valid pixels.

    A pixel is "hard" if its loss is above -log(thresh); at least `min_kept` pixels are always kept.
    """

    def __init__(self, thresh=0.7, min_kept=50_000, weight=None, ignore_index=255):
        super().__init__()
        self.thr = -math.log(thresh)
        self.min_kept = min_kept
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, reduction="none")

    def forward(self, logits, target):
        loss = self.ce(logits, target).reshape(-1)
        loss = loss[target.reshape(-1) != self.ignore_index]
        if loss.numel() == 0:                     # all pixels ignored: zero loss, valid graph
            return logits.sum() * 0.0
        k = min(self.min_kept, loss.numel())
        hard = loss[loss > self.thr]
        if hard.numel() < k:
            hard = loss.topk(k).values
        return hard.mean()


def build_loss(class_weights=None, ohem=False, ignore_index=255):
    w = None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
    if ohem:
        return OHEMCrossEntropy(weight=w, ignore_index=ignore_index)
    return nn.CrossEntropyLoss(weight=w, ignore_index=ignore_index)
