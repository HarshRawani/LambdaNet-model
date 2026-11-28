import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from lambdanet_data.labels import CLASS_NAMES
from lambdanet_data.metrics import SegMetrics


@torch.no_grad()
def evaluate(model, loader, device, amp=True, stride=1, num_classes=19):
    """mIoU / pixel accuracy / ECE on one loader (batch size 1).

    The label is at its ORIGINAL size, so logits are upsampled to it before the metric.
    stride > 1 evaluates every stride-th image (fast checks during training).
    """
    model.eval()
    if stride > 1:
        loader = DataLoader(Subset(loader.dataset, range(0, len(loader.dataset), stride)),
                            batch_size=1, shuffle=False, num_workers=loader.num_workers)
    metrics = SegMetrics(num_classes)
    use_amp = amp and device.type == "cuda"
    for b in loader:
        x = b["image"].to(device, non_blocking=True)
        y = b["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        logits = logits.float()
        if logits.shape[-2:] != y.shape[-2:]:
            logits = F.interpolate(logits, size=y.shape[-2:], mode="bilinear", align_corners=False)
        metrics.update(logits, y)
    out = metrics.compute()
    out["num_images"] = len(loader.dataset)
    return out


def format_results(name, res):
    lines = [f"{name}  ({res['num_images']} images)",
             f"  mIoU {100 * res['mIoU']:.2f}%   pixel acc {100 * res['pixel_acc']:.2f}%   ECE {res['ECE']:.4f}"]
    for n, v in zip(CLASS_NAMES, res["per_class_iou"]):
        lines.append(f"    {n:14s} " + ("   -   (absent)" if v != v else f"{100 * v:6.2f}"))
    return "\n".join(lines)
