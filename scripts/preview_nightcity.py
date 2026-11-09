
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

base = Path("data/NightCity")
image_dir = base / "NightCity-images/images/train"
label_dir = base / "NightCity-label/color/train"

output_dir = Path("check_out/NightCity")
output_dir.mkdir(parents=True, exist_ok=True)

samples = ["Chicago_0002", "Chicago_0005", "Chicago_0008"]

for name in samples:
    image_path = image_dir / f"{name}.png"
    label_path = label_dir / f"{name}_color.png"

    if not image_path.exists():
        print("Missing image:", image_path)
        continue

    if not label_path.exists():
        print("Missing color mask:", label_path)
        continue

    image = Image.open(image_path).convert("RGB")
    mask = Image.open(label_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original NightCity Image")

    axes[1].imshow(mask)
    axes[1].set_title("Ground Truth Color Mask")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    output_path = output_dir / f"{name}_preview.jpg"
    plt.savefig(output_path, dpi=120)
    plt.close()

    print("Saved:", output_path)
