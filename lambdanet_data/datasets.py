import json
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .labels import IGNORE_INDEX, to_train_ids


def list_images(img_dir, suffixes=(".png", ".jpg", ".jpeg")):
    img_dir = Path(img_dir)
    if not img_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {img_dir}")
    return sorted(p for p in img_dir.rglob("*") if p.is_file() and p.name.endswith(tuple(suffixes)))


def find_pairs(img_dir, lbl_dir, img_suffixes, lbl_candidates):
    """Match images to label files by filename stem, wherever they sit in the folders.

    img_suffixes   : e.g. ["_rgb_anon.png"]; stripped from the image name to get the stem.
    lbl_candidates : [(label_suffix, label_mode), ...] tried in order; the first label
                     file named  <stem><label_suffix>  wins. label_mode is "train_id" or "label_id".
    """
    img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)
    for d in (img_dir, lbl_dir):
        if not d.is_dir():
            raise FileNotFoundError(f"directory not found: {d}")

    suffixes = sorted(img_suffixes, key=len, reverse=True)
    lbl_index = {}
    for p in lbl_dir.rglob("*"):
        if p.is_file():
            lbl_index.setdefault(p.name, p)

    pairs, missing = [], 0
    for p in sorted(img_dir.rglob("*")):
        if not p.is_file():
            continue
        suf = next((s for s in suffixes if p.name.endswith(s)), None)
        if suf is None:
            continue
        stem = p.name[: -len(suf)]
        for lsuf, mode in lbl_candidates:
            q = lbl_index.get(stem + lsuf)
            if q is not None:
                pairs.append((p, q, mode))
                break
        else:
            missing += 1
    if missing:
        warnings.warn(f"{missing} images in {img_dir} had no matching label in {lbl_dir}")
    if not pairs:
        raise RuntimeError(f"no image/label pairs found ({img_dir} vs {lbl_dir}); "
                           f"check the suffixes in configs/data.yaml")
    return pairs


def bdd_night_stems(attr_json, timeofday="night"):
    """BDD100K keeps 'timeofday' in the per-image attribute JSON, not in the mask files."""
    with open(attr_json) as f:
        data = json.load(f)
    return {Path(d["name"]).stem for d in data
            if d.get("attributes", {}).get("timeofday") == timeofday}


def bdd_pairs(root, split_dir, attr_json=None, timeofday="night"):
    root = Path(root)
    if attr_json is None:
        cands = [root / "labels" / f"bdd100k_labels_images_{split_dir}.json",
                 root / "labels" / "det_20" / f"det_{split_dir}.json"]
        attr_json = next((c for c in cands if c.exists()), None)
        if attr_json is None:
            raise FileNotFoundError("BDD100K attribute JSON not found; set attr_json in the config. "
                                    f"Tried: {[str(c) for c in cands]}")
    else:
        attr_json = root / attr_json
    keep = bdd_night_stems(attr_json, timeofday)
    pairs = find_pairs(root / "images" / "10k" / split_dir,
                       root / "labels" / "sem_seg" / "masks" / split_dir,
                       [".jpg"], [(".png", "train_id")])
    pairs = [p for p in pairs if p[0].stem in keep]
    if not pairs:
        raise RuntimeError(f"no BDD100K '{timeofday}' segmentation images found for split {split_dir}")
    return pairs


class PairedSegDataset(Dataset):
    def __init__(self, pairs, transform, name="seg", limit=None):
        self.pairs = pairs[:limit] if limit else pairs
        self.transform = transform
        self.name = name

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        ip, lp, mode = self.pairs[i]
        img = Image.open(ip).convert("RGB")
        lbl = np.array(Image.open(lp))
        if lbl.ndim == 3:
            lbl = lbl[..., 0]
        lbl = to_train_ids(lbl, mode)
        if lbl.shape != (img.height, img.width):  # keep image/label aligned
            lbl = np.asarray(Image.fromarray(lbl).resize(img.size, Image.NEAREST))
        image, label = self.transform(img, lbl)
        return {"image": image, "label": label, "name": ip.stem, "dataset": self.name}


class ImageOnlyDataset(Dataset):
    """Unlabelled images (Dark Zurich night/twilight train, Dark Zurich test for server upload).
    The label is all-ignore so the same transforms and loaders can be reused."""

    def __init__(self, files, transform, name="unlabeled", limit=None):
        self.files = files[:limit] if limit else files
        self.transform = transform
        self.name = name

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        p = self.files[i]
        img = Image.open(p).convert("RGB")
        lbl = np.full((img.height, img.width), IGNORE_INDEX, dtype=np.uint8)
        image, label = self.transform(img, lbl)
        return {"image": image, "label": label, "name": p.stem, "dataset": self.name}
