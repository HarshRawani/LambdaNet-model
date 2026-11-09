
from pathlib import Path
from PIL import Image
import numpy as np

root = Path("data/NightCity/NightCity-label/label")

for split in ["train", "val"]:
    files = sorted((root / split).glob("*"))

    if not files:
        print(f"{split}: NO LABELS FOUND")
        continue

    img = Image.open(files[0])
    arr = np.array(img)

    print(f"\n{'='*40}")
    print(f"SPLIT: {split.upper()}")
    print("File:", files[0].name)
    print("Mode:", img.mode)
    print("Shape:", arr.shape)
    print("Data type:", arr.dtype)
    print("Unique values:", np.unique(arr)[:100])
