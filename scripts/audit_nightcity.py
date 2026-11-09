
from pathlib import Path
from collections import Counter
from PIL import Image
import numpy as np

base = Path("data/NightCity")
image_root = base / "NightCity-images/images"
label_root = base / "NightCity-label/label"

# Cityscapes official label IDs
valid_ids = {
    0, 1, 2, 3, 4, 5, 6,
    7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31,
    32, 33
}

for split in ["train", "val"]:
    images = sorted((image_root / split).glob("*.png"))
    labels = sorted((label_root / split).glob("*.png"))

    label_map = {p.stem.removesuffix("_labelIds"): p for p in labels}

    missing = []
    mismatched = []
    unknown = Counter()
    all_values = Counter()

    for img_path in images:
        label_path = label_map.get(img_path.stem)

        if label_path is None:
            missing.append(img_path.name)
            continue

        with Image.open(img_path) as img, Image.open(label_path) as lbl:
            if img.size != lbl.size:
                mismatched.append(img_path.name)

            values, counts = np.unique(
                np.array(lbl), return_counts=True
            )

            for value, count in zip(values, counts):
                all_values[int(value)] += int(count)

                if int(value) not in valid_ids:
                    unknown[int(value)] += int(count)

    print(f"\n{'='*45}")
    print(f"SPLIT: {split.upper()}")
    print("Images:", len(images))
    print("Labels:", len(labels))
    print("Missing labels:", len(missing))
    print("Dimension mismatches:", len(mismatched))
    print("Unknown label IDs:", dict(unknown))
    print("Present label IDs:", sorted(all_values))
    print("Total label pixels:", sum(all_values.values()))
