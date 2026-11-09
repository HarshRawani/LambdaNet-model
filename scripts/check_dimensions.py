
from pathlib import Path
from PIL import Image

base = Path("data/NightCity")

for split in ["train", "val"]:
    image_dir = base / "NightCity-images/images" / split
    label_dir = base / "NightCity-label/label" / split

    image = sorted(image_dir.glob("*.png"))[0]
    label = label_dir / f"{image.stem}_labelIds.png"

    print(f"\n{split.upper()}")
    print("Image:", image.name, Image.open(image).size)
    print("Label:", label.name, Image.open(label).size)
    print("Dimensions match:",
          Image.open(image).size == Image.open(label).size)
