from pathlib import Path

import yaml
from torch.utils.data import ConcatDataset, DataLoader

from .datasets import ImageOnlyDataset, PairedSegDataset, bdd_pairs, find_pairs, list_images
from .transforms import EvalTransform, TrainTransform


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _transform(cfg, train):
    if train:
        t = cfg["train_aug"]
        return TrainTransform(crop_size=tuple(t["crop_size"]), base_height=t["base_height"],
                              scale_range=tuple(t["scale_range"]), hflip=t.get("hflip", True),
                              photometric=t.get("photometric", True))
    return EvalTransform(eval_height=cfg["eval"]["height"])


def build_dataset(cfg, name, split, train=False):
    ds = cfg["datasets"][name]
    root = Path(ds["root"])
    sp = ds["splits"][split]
    tf = _transform(cfg, train)
    limit = cfg.get("limit")  # set a small number for quick debugging runs
    kind = ds.get("kind", "paired")

    if kind == "bdd":
        pairs = bdd_pairs(root, sp.get("split_dir", split), sp.get("attr_json"), ds.get("timeofday", "night"))
        return PairedSegDataset(pairs, tf, name=name, limit=limit)
    if kind == "image_only":
        files = list_images(root / sp["img_dir"], sp.get("img_suffixes", [".png", ".jpg"]))
        return ImageOnlyDataset(files, tf, name=name, limit=limit)

    pairs = find_pairs(root / sp["img_dir"], root / sp["lbl_dir"], sp["img_suffixes"],
                       [tuple(c) for c in sp["lbl_candidates"]])
    return PairedSegDataset(pairs, tf, name=name, limit=limit)


def build_train_loader(cfg, stage="pretrain", batch_size=4, num_workers=4):
    """stage 'pretrain' / 'finetune' are defined under `stages:` in the config."""
    parts = []
    for src in cfg["stages"][stage]:
        ds = build_dataset(cfg, src["dataset"], src.get("split", "train"), train=True)
        parts += [ds] * int(src.get("repeat", 1))
    return DataLoader(ConcatDataset(parts), batch_size=batch_size, shuffle=True,
                      num_workers=num_workers, pin_memory=True, drop_last=True,
                      persistent_workers=num_workers > 0)


def build_eval_loaders(cfg, which=None, num_workers=2):
    """One batch-size-1 loader per entry in `eval_sets`, keyed 'dataset/split'."""
    loaders = {}
    for e in cfg["eval_sets"]:
        key = f'{e["dataset"]}/{e["split"]}'
        if which and key not in which:
            continue
        ds = build_dataset(cfg, e["dataset"], e["split"], train=False)
        loaders[key] = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers)
    return loaders
