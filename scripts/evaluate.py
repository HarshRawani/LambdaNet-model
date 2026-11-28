"""Evaluate a checkpoint on the eval sets in configs/data.yaml.

    python scripts/evaluate.py --ckpt runs/baseline/best.pt --out runs/baseline/eval.json
    python scripts/evaluate.py --ckpt runs/baseline/best.pt --sets ndd/test
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import evaluate, format_results  # noqa: E402
from lambdanet_data import build_eval_loaders, load_config  # noqa: E402
from models import SegBaseline  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="*", default=None, help="e.g. nightcity/val ndd/test (default: all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegBaseline(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    loaders = build_eval_loaders(load_config(args.config), which=args.sets, num_workers=2)
    results = {}
    for name, loader in loaders.items():
        results[name] = evaluate(model, loader, device, amp=not args.no_amp)
        print(format_results(name, results[name]))
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print("saved", args.out)


if __name__ == "__main__":
    main()
