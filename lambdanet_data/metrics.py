import torch


class SegMetrics:
    """Accumulates mIoU, pixel accuracy and Expected Calibration Error (ECE).

    update(logits, target): logits (B, C, H, W) at the SAME size as target (B, H, W).
    ECE here is pixel-level: confidence = max softmax prob, 15 equal-width bins.
    """

    def __init__(self, num_classes=19, ignore_index=255, n_bins=15):
        self.C, self.ignore, self.n_bins = num_classes, ignore_index, n_bins
        self.reset()

    def reset(self):
        self.conf_mat = torch.zeros(self.C, self.C, dtype=torch.float64)
        self.bin_count = torch.zeros(self.n_bins, dtype=torch.float64)
        self.bin_conf = torch.zeros(self.n_bins, dtype=torch.float64)
        self.bin_acc = torch.zeros(self.n_bins, dtype=torch.float64)

    @torch.no_grad()
    def update(self, logits, target):
        conf, pred = logits.softmax(dim=1).max(dim=1)
        valid = target != self.ignore
        t, p, c = target[valid], pred[valid], conf[valid].double()
        self.conf_mat += torch.bincount(t * self.C + p, minlength=self.C ** 2).view(self.C, self.C).double().cpu()
        correct = (p == t).double()
        b = torch.clamp((c * self.n_bins).long(), max=self.n_bins - 1)
        self.bin_count += torch.bincount(b, minlength=self.n_bins).double().cpu()
        self.bin_conf += torch.bincount(b, weights=c, minlength=self.n_bins).cpu()
        self.bin_acc += torch.bincount(b, weights=correct, minlength=self.n_bins).cpu()

    def compute(self):
        cm = self.conf_mat
        tp = cm.diag()
        union = cm.sum(0) + cm.sum(1) - tp
        iou = tp / union.clamp(min=1)
        iou[union == 0] = float("nan")  # class absent from this eval set
        total = cm.sum().clamp(min=1)
        n = self.bin_count.sum().clamp(min=1)
        ece = ((self.bin_acc - self.bin_conf).abs().sum() / n).item()
        return {"mIoU": torch.nanmean(iou).item(), "pixel_acc": (tp.sum() / total).item(),
                "ECE": ece, "per_class_iou": iou.tolist()}
