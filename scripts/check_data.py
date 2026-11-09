"""Verify every dataset in configs/data.yaml: pair counts, label values, class balance,
and a preview image per dataset.

    python scripts/check_data.py --config configs/data.yaml --samples 8 --out check_out
    python scripts/check_data.py --weights 300      # also print class weights from NightCity train
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lambdanet_data import CLASS_NAMES, IGNORE_INDEX, NUM_CLASSES, build_dataset, colorize, load_config  # noqa: E402
from lambdanet_data.transforms import MEAN, STD  # noqa: E402


def show(t):
    img = np.clip(t.permute(1, 2, 0).numpy() * STD + MEAN, 0, 1)
    img = np.power(img, 0.5)  # brighten for display only; night images look black otherwise
    return (img * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--out", default="check_out")
    ap.add_argument("--weights", type=int, default=0, help="samples used to compute class weights")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    targets = []
    for stage in cfg["stages"].values():
        targets += [(s["dataset"], s.get("split", "train")) for s in stage]
    targets += [(e["dataset"], e["split"]) for e in cfg["eval_sets"]]
    targets = list(dict.fromkeys(targets))

    for name, split in targets:
        tag = f"{name}/{split}"
        try:
            ds = build_dataset(cfg, name, split, train=False)
        except Exception as e:  # report and keep going so one bad path doesn't hide the rest
            print(f"[FAIL] {tag}: {type(e).__name__}: {e}")
            continue
        print(f"[ OK ] {tag}: {len(ds)} samples")

        hist = np.zeros(NUM_CLASSES, dtype=np.int64)
        ignored = total = 0
        tiles = []
        for i in np.linspace(0, len(ds) - 1, min(args.samples, len(ds))).astype(int):
            s = ds[int(i)]
            lbl = s["label"].numpy()
            bad = set(np.unique(lbl)) - set(range(NUM_CLASSES)) - {IGNORE_INDEX}
            assert not bad, f"{tag}: unexpected label values {bad}"
            valid = lbl != IGNORE_INDEX
            hist += np.bincount(lbl[valid], minlength=NUM_CLASSES)
            ignored += (~valid).sum()
            total += lbl.size
            if len(tiles) < 4:
                im = Image.fromarray(show(s["image"]))
                lb = Image.fromarray(colorize(lbl)).resize(im.size, Image.NEAREST)
                tiles.append(np.concatenate([np.asarray(im), np.asarray(lb)], axis=0))
        print(f"        ignore pixels: {100 * ignored / max(total, 1):.1f}% | "
              f"present classes: {int((hist > 0).sum())}/{NUM_CLASSES}")
        if tiles:
            h = min(t.shape[0] for t in tiles)
            Image.fromarray(np.concatenate([t[:h] for t in tiles], axis=1)).save(
                out / f"{name}_{split}.jpg", quality=90)

    if args.weights:
        ds = build_dataset(cfg, "nightcity", "train", train=False)
        hist = np.zeros(NUM_CLASSES, dtype=np.float64)
        for i in np.linspace(0, len(ds) - 1, min(args.weights, len(ds))).astype(int):
            lbl = ds[int(i)]["label"].numpy()
            hist += np.bincount(lbl[lbl != IGNORE_INDEX], minlength=NUM_CLASSES)
        freq = hist / max(hist.sum(), 1)
        w = 1.0 / np.log(1.02 + freq)  # ENet-style class weights
        print("\nclass weights (ENet-style) from NightCity train:")
        for n, f, x in zip(CLASS_NAMES, freq, w):
            print(f"  {n:14s} freq={f:.4f}  weight={x:.2f}")


if __name__ == "__main__":
    main()
