
from pathlib import Path
from collections import Counter
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lambdanet_data.labels import (
    to_train_ids,
    NUM_CLASSES,
    IGNORE_INDEX,
)

root = Path("data/NightCity/NightCity-label/label")

for split in ["train", "val"]:
    files = sorted((root / split).glob("*_labelIds.png"))

    raw_counts = Counter()
    train_counts = Counter()

    for path in files:
        raw = np.array(Image.open(path))
        mapped = to_train_ids(raw, "label_id")

        raw_ids, raw_pixels = np.unique(raw, return_counts=True)
        train_ids, train_pixels = np.unique(mapped, return_counts=True)

        raw_counts.update(dict(zip(raw_ids.tolist(), raw_pixels.tolist())))
        train_counts.update(dict(zip(train_ids.tolist(), train_pixels.tolist())))

    total_pixels = sum(train_counts.values())
    ignored_pixels = train_counts[IGNORE_INDEX]
    valid_pixels = total_pixels - ignored_pixels

    print(f"\n--- {split.upper()} ---")
    print("Files:", len(files))
    print("Total pixels:", f"{total_pixels:,}")
    print("Raw IDs:", sorted(raw_counts))
    print("Train IDs:", sorted(train_counts))
    print("Valid classes:", sorted(k for k in train_counts if k < NUM_CLASSES))
    print("Ignored pixels:", f"{ignored_pixels:,}")
    print("Valid pixels:", f"{valid_pixels:,}")
    print("Ignore percentage:", f"{ignored_pixels / total_pixels * 100:.6f}%")
