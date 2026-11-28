"""Train the Phase 2 baseline.

    python scripts/train.py --stage pretrain --epochs 30 --out runs/baseline
    python scripts/train.py --stage pretrain --class-weights class_weights.json --ohem --out runs/baseline_w
    python scripts/train.py --resume runs/baseline/last.pt --out runs/baseline

Defaults suit a 4 GB GPU: batch 2 x gradient accumulation 4 (effective 8), mixed precision.
Model selection uses --select-on (default nightcity/val) only. NDD / Dark Zurich are never used for tuning.
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import evaluate, format_results  # noqa: E402
from lambdanet_data import build_eval_loaders, build_train_loader, load_config  # noqa: E402
from losses import build_loss  # noqa: E402
from models import SegBaseline  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--stage", default="pretrain")
    ap.add_argument("--out", default="runs/baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.5)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--warmup", type=int, default=300, help="warm-up optimizer steps")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--unfreeze-bn", action="store_true")
    ap.add_argument("--class-weights", default=None, help="json from compute_class_weights.py")
    ap.add_argument("--ohem", action="store_true")
    ap.add_argument("--eval-every", type=int, default=2, help="epochs between validation runs")
    ap.add_argument("--val-stride", type=int, default=4, help="evaluate every k-th val image during training")
    ap.add_argument("--select-on", default="nightcity/val")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--limit", type=int, default=None, help="debug: use only N samples per dataset")
    ap.add_argument("--resume", default=None)
    return ap.parse_args()


def main():
    args = get_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.config)
    if args.limit:
        cfg["limit"] = args.limit
    train_loader = build_train_loader(cfg, args.stage, args.batch_size, args.workers)
    eval_loaders = build_eval_loaders(cfg, num_workers=2)
    if args.select_on not in eval_loaders:
        sys.exit(f"--select-on {args.select_on} not in eval sets {list(eval_loaders)}")

    model = SegBaseline(pretrained=not args.no_pretrained, freeze_bn=not args.unfreeze_bn).to(device)
    weights = json.load(open(args.class_weights))["weights"] if args.class_weights else None
    criterion = build_loss(weights, ohem=args.ohem).to(device)
    optimizer = AdamW(model.param_groups(args.lr, args.wd, args.backbone_lr_mult))

    steps_per_epoch = max(len(train_loader) // args.accum, 1)
    total_steps = steps_per_epoch * args.epochs
    warmup = max(min(args.warmup, total_steps // 10), 1)

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        t = (step - warmup) / max(1, total_steps - warmup)
        return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best, global_step = 0, -1.0, 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"]); scaler.load_state_dict(ck["scaler"])
        start_epoch, best, global_step = ck["epoch"] + 1, ck["best"], ck["global_step"]
        print(f"resumed from {args.resume} at epoch {start_epoch}")

    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out / "tb")
    except Exception:
        writer = None

    print(f"device {device} | amp {use_amp} | train samples {len(train_loader.dataset)} | "
          f"{len(train_loader)} batches/epoch | effective batch {args.batch_size * args.accum}")
    log = open(out / "log.jsonl", "a")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        run_loss, n, t0 = 0.0, 0, time.time()
        for i, b in enumerate(train_loader):
            x = b["image"].to(device, non_blocking=True)
            y = b["label"].to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            loss = criterion(logits.float(), y)
            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step(); global_step += 1
            run_loss += loss.item(); n += 1
            if (i + 1) % args.log_every == 0:
                print(f"epoch {epoch + 1}/{args.epochs} iter {i + 1}/{len(train_loader)} "
                      f"loss {run_loss / n:.4f} lr {scheduler.get_last_lr()[-1]:.2e} "
                      f"{(time.time() - t0) / n:.2f}s/iter")
                if writer:
                    writer.add_scalar("train/loss", run_loss / n, global_step)
        rec = {"epoch": epoch + 1, "train_loss": run_loss / max(n, 1), "epoch_time_s": time.time() - t0}
        if device.type == "cuda":
            rec["peak_gpu_gb"] = torch.cuda.max_memory_allocated() / 2 ** 30
        print(f"== epoch {epoch + 1} done: loss {rec['train_loss']:.4f}, {rec['epoch_time_s'] / 60:.1f} min"
              + (f", peak GPU memory {rec['peak_gpu_gb']:.2f} GB" if "peak_gpu_gb" in rec else ""))

        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            res = evaluate(model, eval_loaders[args.select_on], device, use_amp, stride=args.val_stride)
            rec["val_mIoU"], rec["val_pixel_acc"], rec["val_ECE"] = res["mIoU"], res["pixel_acc"], res["ECE"]
            print(f"   {args.select_on} (every {args.val_stride}th image): mIoU {100 * res['mIoU']:.2f}% "
                  f"acc {100 * res['pixel_acc']:.2f}% ECE {res['ECE']:.4f}")
            if writer:
                writer.add_scalar("val/mIoU", res["mIoU"], epoch + 1)
            if res["mIoU"] > best:
                best = res["mIoU"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "best": best, "args": vars(args)},
                           out / "best.pt")
                print("   new best -> best.pt")
        log.write(json.dumps(rec) + "\n"); log.flush()
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                    "epoch": epoch, "best": best, "global_step": global_step, "args": vars(args)},
                   out / "last.pt")

    # final full evaluation of the best checkpoint on every eval set
    if (out / "best.pt").exists():
        model.load_state_dict(torch.load(out / "best.pt", map_location=device, weights_only=False)["model"])
    results = {}
    for name, loader in eval_loaders.items():
        results[name] = evaluate(model, loader, device, use_amp)
        print(format_results(name, results[name]))
    json.dump(results, open(out / "final_results.json", "w"), indent=2)
    print("saved", out / "final_results.json")


if __name__ == "__main__":
    main()
