"""Smoke test with tiny fake data (no real datasets needed):  python tests/test_smoke.py"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lambdanet_data import SegMetrics, build_eval_loaders, build_train_loader  # noqa: E402


def save_pair(img_p, lbl_p, ids):
    img_p.parent.mkdir(parents=True, exist_ok=True)
    lbl_p.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.random.randint(0, 40, (100, 200, 3), dtype=np.uint8)).save(img_p)
    Image.fromarray(np.random.choice(ids, (100, 200)).astype(np.uint8)).save(lbl_p)


def main():
    tmp = Path(tempfile.mkdtemp())
    label_ids = [0, 7, 8, 26, 33]      # includes 0 -> ignore
    train_ids = [0, 1, 13, 18, 255]
    names = {"train": [], "val": []}
    for sp in ("train", "val"):
        for i in range(4):
            save_pair(tmp / f"nc/images/{sp}/a{i}.png", tmp / f"nc/labels/{sp}/a{i}_labelIds.png", label_ids)
            save_pair(tmp / f"bdd/images/10k/{sp}/b{i}.jpg", tmp / f"bdd/labels/sem_seg/masks/{sp}/b{i}.png", train_ids)
            names[sp].append({"name": f"b{i}.jpg", "attributes": {"timeofday": "night" if i % 2 == 0 else "daytime"}})
        (tmp / "bdd/labels").mkdir(parents=True, exist_ok=True)
        (tmp / f"bdd/labels/bdd100k_labels_images_{sp}.json").write_text(json.dumps(names[sp]))
    for i in range(3):
        save_pair(tmp / f"dz/rgb_anon/val/night/G/f{i}_rgb_anon.png", tmp / f"dz/gt/val/night/G/f{i}_gt_labelTrainIds.png", train_ids)
        save_pair(tmp / f"ndd/leftImg8bit/test/night/n{i}_leftImg8bit.png",
                  tmp / f"ndd/gtCoarse/test/night/n{i}_gtCoarse_labelIds.png", label_ids)

    cfg = {
        "limit": None,
        "train_aug": {"crop_size": [64, 64], "base_height": 64, "scale_range": [0.75, 1.5]},
        "eval": {"height": 64},
        "stages": {"finetune": [{"dataset": "nightcity", "split": "train"},
                                {"dataset": "bdd100k_night", "split": "train", "repeat": 2}]},
        "eval_sets": [{"dataset": d, "split": s} for d, s in
                      [("nightcity", "val"), ("bdd100k_night", "val"), ("dark_zurich", "val"), ("ndd", "test")]],
        "datasets": {
            "nightcity": {"root": str(tmp / "nc"), "splits": {
                sp: {"img_dir": f"images/{sp}", "lbl_dir": f"labels/{sp}", "img_suffixes": [".png"],
                     "lbl_candidates": [["_labelIds.png", "label_id"]]} for sp in ("train", "val")}},
            "bdd100k_night": {"kind": "bdd", "root": str(tmp / "bdd"),
                              "splits": {"train": {"split_dir": "train"}, "val": {"split_dir": "val"}}},
            "dark_zurich": {"root": str(tmp / "dz"), "splits": {"val": {
                "img_dir": "rgb_anon/val/night", "lbl_dir": "gt/val/night", "img_suffixes": ["_rgb_anon.png"],
                "lbl_candidates": [["_gt_labelTrainIds.png", "train_id"]]}}},
            "ndd": {"root": str(tmp / "ndd"), "splits": {"test": {
                "img_dir": "leftImg8bit/test/night", "lbl_dir": "gtCoarse/test/night",
                "img_suffixes": ["_leftImg8bit.png"], "lbl_candidates": [["_gtCoarse_labelIds.png", "label_id"]]}}},
        },
    }

    loader = build_train_loader(cfg, "finetune", batch_size=2, num_workers=0)
    b = next(iter(loader))
    assert b["image"].shape == (2, 3, 64, 64) and b["label"].shape == (2, 64, 64)
    assert set(b["label"].unique().tolist()) <= set(range(19)) | {255}
    print("train batch OK:", tuple(b["image"].shape), "| total train samples:", len(loader.dataset))

    # BDD night filter should keep only 2 of 4 images per split
    ev = build_eval_loaders(cfg, num_workers=0)
    assert len(ev["bdd100k_night/val"].dataset) == 2
    for key, dl in ev.items():
        s = next(iter(dl))
        print(f"eval {key}: {len(dl.dataset)} samples, image {tuple(s['image'].shape)}, label {tuple(s['label'].shape)}")

    # metrics: perfect prediction -> mIoU 1, acc 1; ECE equals 1 - confidence for one-hot-ish logits
    m = SegMetrics()
    tgt = torch.randint(0, 19, (1, 32, 32))
    logits = torch.nn.functional.one_hot(tgt, 19).permute(0, 3, 1, 2).float() * 20
    m.update(logits, tgt)
    r = m.compute()
    assert abs(r["pixel_acc"] - 1) < 1e-9 and r["ECE"] < 1e-3, r
    print("metrics OK:", {k: round(v, 4) for k, v in r.items() if k != "per_class_iou"})
    print("ALL OK")


if __name__ == "__main__":
    main()
