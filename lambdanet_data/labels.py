"""Shared 19-class Cityscapes label space used by every dataset in this project.

NightCity, Dark Zurich, Nighttime Driving and BDD100K all use (or map to) the
19 Cityscapes "train" classes, so one mapping covers everything.
Pixels that are not one of the 19 classes get IGNORE_INDEX and are skipped by the
loss and the metrics.
"""
import numpy as np

IGNORE_INDEX = 255
NUM_CLASSES = 19

CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]

PALETTE = np.array([
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32),
], dtype=np.uint8)

# Cityscapes labelId -> trainId (all other ids -> IGNORE_INDEX)
_LABEL_ID_TO_TRAIN_ID = {
    7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 31: 16, 32: 17, 33: 18,
}
LABEL_ID_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _k, _v in _LABEL_ID_TO_TRAIN_ID.items():
    LABEL_ID_LUT[_k] = _v


def to_train_ids(mask: np.ndarray, mode: str) -> np.ndarray:
    """Convert a raw label mask to trainIds (0..18, 255 = ignore).

    mode: "train_id" (mask already holds trainIds) or "label_id" (Cityscapes labelIds).
    """
    mask = mask.astype(np.uint8)
    if mode == "train_id":
        out = mask.copy()
        out[(out >= NUM_CLASSES) & (out != IGNORE_INDEX)] = IGNORE_INDEX
        return out
    if mode == "label_id":
        return LABEL_ID_LUT[mask]
    raise ValueError(f"unknown label mode: {mode}")


def colorize(mask: np.ndarray) -> np.ndarray:
    """trainId mask (H, W) -> RGB uint8 (H, W, 3); ignore pixels are black."""
    out = np.zeros(mask.shape + (3,), dtype=np.uint8)
    valid = mask < NUM_CLASSES
    out[valid] = PALETTE[mask[valid]]
    return out
