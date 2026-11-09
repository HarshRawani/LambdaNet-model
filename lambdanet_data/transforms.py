"""Paired image/label transforms (PIL + numpy + torch only).

Images are resized by HEIGHT while keeping aspect ratio, because the datasets have
different shapes (Cityscapes-style 2:1, BDD 16:9, Dark Zurich 16:9).

Training augmentation is deliberately mild on brightness: night images are already
dark, and aggressive jitter can wipe out the weak signal the model must learn from.
"""
import random

import numpy as np
import torch
from PIL import Image

from .labels import IGNORE_INDEX

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _normalize(img01: np.ndarray) -> torch.Tensor:
    arr = (img01 - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1))).float()


class TrainTransform:
    def __init__(self, crop_size=(512, 512), base_height=512, scale_range=(0.75, 1.5),
                 hflip=True, photometric=True):
        self.crop_h, self.crop_w = crop_size
        self.base_height = base_height
        self.scale_range = scale_range
        self.hflip = hflip
        self.photometric = photometric

    def __call__(self, img: Image.Image, lbl: np.ndarray):
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        new_h = max(int(self.base_height * scale), 1)
        new_w = max(int(round(w * new_h / h)), 1)
        img = np.asarray(img.resize((new_w, new_h), Image.BILINEAR))
        lbl = np.asarray(Image.fromarray(lbl).resize((new_w, new_h), Image.NEAREST))

        pad_h, pad_w = max(self.crop_h - new_h, 0), max(self.crop_w - new_w, 0)
        if pad_h or pad_w:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), constant_values=0)
            lbl = np.pad(lbl, ((0, pad_h), (0, pad_w)), constant_values=IGNORE_INDEX)

        H, W = lbl.shape
        y = random.randint(0, H - self.crop_h)
        x = random.randint(0, W - self.crop_w)
        img = img[y:y + self.crop_h, x:x + self.crop_w]
        lbl = lbl[y:y + self.crop_h, x:x + self.crop_w]

        if self.hflip and random.random() < 0.5:
            img, lbl = img[:, ::-1], lbl[:, ::-1]

        img = img.astype(np.float32) / 255.0
        if self.photometric:
            img = np.power(img, random.uniform(0.8, 1.25))      # mild gamma
            img = np.clip(img * random.uniform(0.8, 1.2), 0.0, 1.0)  # mild gain

        return _normalize(img), torch.from_numpy(np.ascontiguousarray(lbl)).long()


class EvalTransform:
    """Resize image to a fixed height (width rounded to a multiple of 32).

    The label is returned at its ORIGINAL size, so the eval loop should upsample the
    model's logits to the label size before computing metrics (batch size 1).
    """

    def __init__(self, eval_height=512, multiple=32):
        self.eval_height = eval_height
        self.multiple = multiple

    def __call__(self, img: Image.Image, lbl: np.ndarray):
        w, h = img.size
        new_h = self.eval_height
        new_w = max(int(round(w * new_h / h / self.multiple)) * self.multiple, self.multiple)
        img = np.asarray(img.resize((new_w, new_h), Image.BILINEAR)).astype(np.float32) / 255.0
        return _normalize(img), torch.from_numpy(np.ascontiguousarray(lbl)).long()
