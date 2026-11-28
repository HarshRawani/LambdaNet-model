"""Debug test: the baseline must be able to overfit a handful of images.

    python scripts/overfit_test.py            # 8 NightCity images, 300 iterations
If the loss does not fall clearly, something is wrong (labels, LR, model) -- fix it before a real run.
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lambdanet_data import build_train_loader, load_config  # noqa: E402
from losses import build_loss  # noqa: E402
from models import SegBaseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--images", type=int, default=8)
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=5e-4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.config)
    cfg["limit"] = args.images
    loader = build_train_loader(cfg, "pretrain", batch_size=2, num_workers=0)
    model = SegBaseline().to(device).train()
    criterion = build_loss().to(device)
    opt = torch.optim.AdamW(model.param_groups(args.lr, 0.0, 1.0))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    losses, it = [], 0
    while it < args.iters:
        for b in loader:
            x, y = b["image"].to(device), b["label"].to(device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(x)
            loss = criterion(logits.float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(loss.item()); it += 1
            if it % 25 == 0:
                print(f"iter {it:4d}  loss {sum(losses[-25:]) / 25:.4f}")
            if it >= args.iters:
                break
    first, last = sum(losses[:20]) / 20, sum(losses[-20:]) / 20
    print(f"first-20 mean {first:.4f}  last-20 mean {last:.4f}")
    if last < 0.5 * first:
        print("OVERFIT TEST: PASS")
    else:
        print("OVERFIT TEST: FAIL -- loss did not fall by half")
        sys.exit(1)


if __name__ == "__main__":
    main()
