"""Phase 2 baseline: ImageNet-pretrained ResNet-18 encoder + pyramid-pooling context + FPN decoder.

Why not a dilated ResNet-18? torchvision's BasicBlock (ResNet-18/34) does not support dilation,
so context is added with a pyramid-pooling module instead. The encoder features (c2..c5) and the
context output are exposed via `return_features=True`, so the same code can later serve as
LambdaNet's *semantic branch*.

Small-batch friendly: encoder BatchNorm layers are frozen (ImageNet statistics), and every new
layer uses GroupNorm, so nothing depends on the batch size.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ConvGNReLU(nn.Sequential):
    def __init__(self, cin, cout, k=3):
        super().__init__(
            nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
            nn.GroupNorm(16, cout),
            nn.ReLU(inplace=True),
        )


class PPM(nn.Module):
    """Pyramid pooling: global + regional context at the deepest (stride-32) features."""

    def __init__(self, cin, cout, bins=(1, 2, 3, 6), branch_ch=64):
        super().__init__()
        self.stages = nn.ModuleList(
            [nn.Sequential(nn.AdaptiveAvgPool2d(b), ConvGNReLU(cin, branch_ch, k=1)) for b in bins]
        )
        self.bottleneck = ConvGNReLU(cin + branch_ch * len(bins), cout, k=3)

    def forward(self, x):
        h, w = x.shape[-2:]
        outs = [x] + [F.interpolate(s(x), size=(h, w), mode="bilinear", align_corners=False) for s in self.stages]
        return self.bottleneck(torch.cat(outs, dim=1))


def _up(x, ref):
    return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)


class SegBaseline(nn.Module):
    def __init__(self, num_classes=19, pretrained=True, width=128, freeze_bn=True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        r = torchvision.models.resnet18(weights=weights)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)       # stride 4
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4
        self.freeze_bn = freeze_bn

        self.context = PPM(512, width)
        self.lat4 = nn.Conv2d(256, width, 1)
        self.lat3 = nn.Conv2d(128, width, 1)
        self.lat2 = nn.Conv2d(64, width, 1)
        self.smooth4, self.smooth3, self.smooth2 = (ConvGNReLU(width, width) for _ in range(3))
        self.fuse = ConvGNReLU(width * 4, width)
        self.classifier = nn.Conv2d(width, num_classes, 1)

        if freeze_bn:
            for m in self._encoder_modules():
                for mod in m.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        for p in mod.parameters():
                            p.requires_grad_(False)

    def _encoder_modules(self):
        return [self.stem, self.layer1, self.layer2, self.layer3, self.layer4]

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn:           # keep ImageNet BN statistics fixed
            for m in self._encoder_modules():
                for mod in m.modules():
                    if isinstance(mod, nn.BatchNorm2d):
                        mod.eval()
        return self

    def encode(self, x):
        c1 = self.stem(x)
        c2 = self.layer1(c1)   # stride 4,  64 ch
        c3 = self.layer2(c2)   # stride 8, 128 ch
        c4 = self.layer3(c3)   # stride 16, 256 ch
        c5 = self.layer4(c4)   # stride 32, 512 ch
        return c2, c3, c4, c5

    def forward(self, x, return_features=False):
        size = x.shape[-2:]
        c2, c3, c4, c5 = self.encode(x)
        p5 = self.context(c5)
        p4 = self.smooth4(self.lat4(c4) + _up(p5, c4))
        p3 = self.smooth3(self.lat3(c3) + _up(p4, c3))
        p2 = self.smooth2(self.lat2(c2) + _up(p3, c2))
        fused = self.fuse(torch.cat([p2, _up(p3, p2), _up(p4, p2), _up(p5, p2)], dim=1))
        logits = F.interpolate(self.classifier(fused), size=size, mode="bilinear", align_corners=False)
        if return_features:
            return logits, {"c2": c2, "c3": c3, "c4": c4, "c5": c5, "context": p5, "fused": fused}
        return logits

    def param_groups(self, lr, weight_decay=1e-2, backbone_lr_mult=0.5):
        """AdamW groups: lower LR for the pretrained encoder, no weight decay on norms/biases."""
        backbone_ids = {id(p) for m in self._encoder_modules() for p in m.parameters()}
        buckets = {}
        for p in self.parameters():
            if not p.requires_grad:
                continue
            key = ("backbone" if id(p) in backbone_ids else "head", "decay" if p.ndim > 1 else "no_decay")
            buckets.setdefault(key, []).append(p)
        return [
            {"params": ps,
             "lr": lr * (backbone_lr_mult if part == "backbone" else 1.0),
             "weight_decay": weight_decay if kind == "decay" else 0.0}
            for (part, kind), ps in buckets.items()
        ]
