from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from lambdanet_data.labels import to_train_ids, CLASS_NAMES

DATASETS = {
    "NightCity train": (
        Path("data/NightCity/NightCity-label/label/train"),
        "*_labelIds.png",
        "label_id",
    ),
    "NightCity val": (
        Path("data/NightCity/NightCity-label/label/val"),
        "*_labelIds.png",
        "label_id",
    ),
    "BDD100K train": (
        Path("data/BDD100K_night/masks/train"),
        "*.png",
        "train_id",
    ),
    "BDD100K val": (
        Path("data/BDD100K_night/masks/val"),
        "*.png",
        "train_id",
    ),
    "NDD test": (
        Path("data/NDD/gtCoarse_daytime_trainvaltest/test/night"),
        "*_gtCoarse_labelTrainIds.png",
        "train_id",
    ),
}

for name, (directory, pattern, encoding) in DATASETS.items():
    files = sorted(directory.glob(pattern))
    counts = Counter()
    total_pixels = 0

    for path in files:
        mask = np.asarray(Image.open(path))
        if encoding == "label_id":
            mask = to_train_ids(mask, "label_id")

        values, frequencies = np.unique(mask, return_counts=True)
        counts.update(dict(zip(values.tolist(), frequencies.tolist())))
        total_pixels += mask.size

    ignored = counts.get(255, 0)
    valid = sum(counts.get(i, 0) for i in range(19))

    print(f"\n{'=' * 55}")
    print(f"{name}: {len(files)} masks")
    print(f"Total pixels: {total_pixels:,}")
    print(f"Valid pixels: {valid:,} ({valid / total_pixels * 100:.2f}%)")
    print(f"Ignore pixels: {ignored:,} ({ignored / total_pixels * 100:.2f}%)")

    invalid = {k: v for k, v in counts.items() if k not in range(19) and k != 255}
    print(f"Unexpected IDs: {invalid if invalid else 'None'}")
    print("\nClass distribution (percentage of valid pixels):")

    for i in range(19):
        pixels = counts.get(i, 0)
        percentage = pixels / valid * 100 if valid else 0
        print(f"  {i:2d} {CLASS_NAMES[i]:16s} {pixels:>14,}  {percentage:6.2f}%")