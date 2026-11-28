"""Pixel frequencies of the NightCity train labels -> ENet-style class weights (class_weights.json).

    python scripts/compute_class_weights.py --config configs/data.yaml --out class_weights.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lambdanet_data import CLASS_NAMES, NUM_CLASSES, build_dataset, load_config  # noqa: E402
from lambdanet_data.labels import to_train_ids  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--dataset", default="nightcity")
    ap.add_argument("--out", default="class_weights.json")
    ap.add_argument("--max-weight", type=float, default=30.0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["limit"] = None
    ds = build_dataset(cfg, args.dataset, "train")
    hist = np.zeros(NUM_CLASSES, dtype=np.int64)
    for i, (_, lbl_path, mode) in enumerate(ds.pairs):
        m = to_train_ids(np.array(Image.open(lbl_path)), mode)
        hist += np.bincount(m[m < NUM_CLASSES], minlength=NUM_CLASSES)
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ds.pairs)}")
    freq = hist / hist.sum()
    w = np.minimum(1.0 / np.log(1.02 + freq), args.max_weight)
    json.dump({"source": f"{args.dataset}/train", "freq": freq.tolist(), "weights": w.tolist()},
              open(args.out, "w"), indent=2)
    for n, f, x in zip(CLASS_NAMES, freq, w):
        print(f"  {n:14s} freq={100 * f:6.2f}%  weight={x:5.2f}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
