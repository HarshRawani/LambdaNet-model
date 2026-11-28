"""Model/loss unit tests -- no datasets needed.   python tests/test_baseline.py"""
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from losses import OHEMCrossEntropy, build_loss  # noqa: E402
from models import SegBaseline  # noqa: E402


def main():
    torch.manual_seed(0)
    model = SegBaseline(pretrained=False).train()
    x = torch.randn(2, 3, 128, 192)

    logits, feats = model(x, return_features=True)
    assert logits.shape == (2, 19, 128, 192), logits.shape
    assert feats["c2"].shape[-2:] == (32, 48) and feats["c5"].shape[-2:] == (4, 6)
    assert feats["fused"].shape == (2, 128, 32, 48)
    print("forward shapes OK:", tuple(logits.shape))

    # frozen encoder BN: params fixed and layers kept in eval mode by .train()
    bns = [m for m in model.stem.modules() if isinstance(m, nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)
    assert all(not p.requires_grad for m in bns for p in m.parameters())
    print("frozen BN OK")

    # parameter groups cover every trainable parameter exactly once
    groups = model.param_groups(1e-3)
    n_grouped = sum(p.numel() for g in groups for p in g["params"])
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_grouped == n_trainable
    print("param groups OK:", len(groups), "groups,", n_trainable, "trainable params")

    y = torch.randint(0, 19, (2, 128, 192)); y[:, :10] = 255
    weights = torch.linspace(1, 5, 19)
    for name, crit in [("CE", build_loss()), ("weighted CE", build_loss(weights)),
                       ("OHEM", build_loss(weights, ohem=True)), ("OHEM small", OHEMCrossEntropy(min_kept=100))]:
        model.zero_grad()
        loss = crit(logits.float(), y)
        loss.backward(retain_graph=True)
        assert torch.isfinite(loss), name
        print(f"loss {name:12s} {loss.item():.4f}")
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier.parameters())

    all_ignore = torch.full_like(y, 255)
    assert float(build_loss(ohem=True)(logits.float(), all_ignore)) == 0.0
    print("all-ignore target OK")
    print("ALL BASELINE TESTS PASSED")


if __name__ == "__main__":
    main()
