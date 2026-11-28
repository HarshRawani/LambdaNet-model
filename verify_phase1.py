#!/usr/bin/env python3
"""
LambdaNet Phase 1 — Complete Verification Suite

Run from the repository root (with the project virtual environment active):
    python verify_phase1.py

The repository root is detected from this file's location, so the runner also
works if it is placed inside the scripts/ directory.

This runner executes the Phase 1 checks in sequence, prints each section's
output, continues after a failed section, and reports a final PASS/FAIL table.

Important:
- This verifies the environment and data pipeline, not model accuracy.
- The real-data loader checks validate representative batches, not every
  sample in every loader.
- NDD remains a held-out test set; this script only checks that its loader
  and labels are readable.
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable

SCRIPT_DIR = Path(__file__).resolve().parent

# Support both locations:
#   <repo>/verify_phase1.py
#   <repo>/scripts/verify_phase1.py
_candidates = (SCRIPT_DIR, SCRIPT_DIR.parent)
ROOT = next(
    (candidate for candidate in _candidates if (candidate / "configs" / "data.yaml").is_file()),
    None,
)
if ROOT is None:
    searched = ", ".join(str(p) for p in _candidates)
    raise FileNotFoundError(
        "Could not locate the LambdaNet repository root: expected "
        "configs/data.yaml in one of these directories: " + searched
    )

os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

RESULTS: list[tuple[str, bool, str]] = []


def section(title: str) -> None:
    print("\n" + "=" * 78, flush=True)
    print(title, flush=True)
    print("=" * 78, flush=True)


def record(name: str, fn: Callable[[], None]) -> None:
    section(name)
    try:
        fn()
    except Exception as exc:
        RESULTS.append((name, False, f"{type(exc).__name__}: {exc}"))
        print(f"\n[FAIL] {name}: {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
    else:
        RESULTS.append((name, True, ""))
        print(f"\n[PASS] {name}", flush=True)


def run_command(args: list[str], label: str | None = None) -> None:
    display = label or " ".join(args)
    print(f"\n$ {display}", flush=True)
    completed = subprocess.run(args, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command exited with status {completed.returncode}: {display}"
        )


def test_environment() -> None:
    import numpy
    import PIL
    import torch
    import torchvision
    import yaml

    print("Python:", sys.version.replace("\n", " "))
    print("PyTorch:", torch.__version__)
    print("Torchvision:", torchvision.__version__)
    print("NumPy:", numpy.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA build:", torch.version.cuda)
    print(
        "GPU:",
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Not available",
    )
    print("Pillow:", PIL.__version__)
    print("PyYAML:", yaml.__version__)
    assert torch.__version__ and torchvision.__version__
    run_command([sys.executable, "-m", "pip", "check"])


def test_project_imports_and_config() -> None:
    from lambdanet_data.build import (
        build_dataset,
        build_eval_loaders,
        build_train_loader,
        load_config,
    )
    from lambdanet_data.datasets import (
        ImageOnlyDataset,
        PairedSegDataset,
        find_pairs,
        list_images,
    )
    from lambdanet_data.labels import (
        CLASS_NAMES,
        IGNORE_INDEX,
        NUM_CLASSES,
        colorize,
        to_train_ids,
    )
    from lambdanet_data.metrics import SegMetrics
    from lambdanet_data.transforms import EvalTransform, TrainTransform

    assert NUM_CLASSES == 19, NUM_CLASSES
    assert IGNORE_INDEX == 255, IGNORE_INDEX
    assert len(CLASS_NAMES) == 19, len(CLASS_NAMES)

    cfg = load_config("configs/data.yaml")
    print("Project imports: OK")
    print("NUM_CLASSES:", NUM_CLASSES)
    print("IGNORE_INDEX:", IGNORE_INDEX)
    print("Datasets:", list(cfg["datasets"]))
    print("Stages:", list(cfg["stages"]))
    print("Evaluation sets:", cfg["eval_sets"])

    expected = {"nightcity", "bdd100k_night", "ndd", "dark_zurich_unlabeled"}
    assert expected.issubset(set(cfg["datasets"]))


def test_dataset_structure() -> None:
    run_command(
        [
            sys.executable,
            "scripts/check_data.py",
            "--config",
            "configs/data.yaml",
            "--samples",
            "8",
        ]
    )


def test_image_dimensions() -> None:
    run_command([sys.executable, "scripts/check_dimensions.py"])


def test_label_mapping() -> None:
    run_command([sys.executable, "scripts/verify_label_mapping.py"])


def test_nightcity_integrity() -> None:
    run_command([sys.executable, "scripts/audit_nightcity.py"])


def test_class_distribution() -> None:
    run_command([sys.executable, "scripts/audit_class_distribution.py"])


def test_metrics_unit() -> None:
    import torch
    from lambdanet_data.metrics import SegMetrics

    metric = SegMetrics(num_classes=19, ignore_index=255)
    target = torch.tensor([[[0, 1, 2, 255]]], dtype=torch.long)
    logits = torch.full((1, 19, 1, 4), -10.0)
    logits[0, 0, 0, 0] = 10.0
    logits[0, 1, 0, 1] = 10.0
    logits[0, 2, 0, 2] = 10.0
    logits[0, 3, 0, 3] = 10.0

    metric.update(logits, target)
    result = metric.compute()
    print("Metric result:", result)

    assert abs(result["mIoU"] - 1.0) < 1e-6, result["mIoU"]
    assert abs(result["pixel_acc"] - 1.0) < 1e-6, result["pixel_acc"]
    assert abs(result["ECE"]) < 1e-6, result["ECE"]
    # Class 3 is present only as a prediction on an ignored target pixel.
    # Classes absent from the valid target set should have NaN IoU.
    assert math.isnan(result["per_class_iou"][3])
    print("Perfect prediction and ignore-index behaviour: verified")


def check_batch(name: str, batch: dict) -> None:
    import torch

    assert isinstance(batch, dict), f"{name}: expected a dictionary batch"
    required = {"image", "label", "name", "dataset"}
    assert required.issubset(batch.keys()), f"{name}: missing keys {required - set(batch)}"

    images = batch["image"]
    labels = batch["label"]

    print(f"\n{name}")
    print("  Image shape:", tuple(images.shape))
    print("  Label shape:", tuple(labels.shape))
    print("  Image dtype:", images.dtype)
    print("  Label dtype:", labels.dtype)
    print("  Image finite:", bool(torch.isfinite(images).all()))
    print("  Label IDs:", torch.unique(labels).tolist())

    assert images.ndim == 4 and images.shape[1] == 3
    assert labels.ndim == 3 and images.shape[0] == labels.shape[0]
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert torch.isfinite(images).all()
    assert ((labels == 255) | ((labels >= 0) & (labels < 19))).all()
    assert len(batch["name"]) == images.shape[0]
    assert len(batch["dataset"]) == images.shape[0]


def test_real_train_loader() -> None:
    from lambdanet_data.build import build_train_loader, load_config

    cfg = load_config("configs/data.yaml")
    loader = build_train_loader(
        cfg, stage="pretrain", batch_size=2, num_workers=0
    )
    batch = next(iter(loader))
    check_batch("Real pretrain training batch", batch)


def test_real_eval_loaders() -> None:
    from lambdanet_data.build import build_eval_loaders, load_config

    cfg = load_config("configs/data.yaml")
    loaders = build_eval_loaders(cfg, num_workers=0)
    print("Evaluation loaders:", list(loaders))

    expected = {"nightcity/val", "bdd100k_night/val", "ndd/test"}
    assert expected.issubset(set(loaders)), (
        f"Missing expected evaluation loaders: {expected - set(loaders)}"
    )

    for name, loader in loaders.items():
        batch = next(iter(loader))
        check_batch(name, batch)


def test_dark_zurich_inference() -> None:
    import torch
    from lambdanet_data.build import build_dataset, load_config

    cfg = load_config("configs/data.yaml")
    dataset = build_dataset(
        cfg, "dark_zurich_unlabeled", "test", train=False
    )
    print("Dark Zurich image count:", len(dataset))
    assert len(dataset) == 151, (
        f"Expected 151 images from the audited archive, found {len(dataset)}"
    )

    sample = dataset[0]
    assert isinstance(sample, dict)
    check_batch(
        "Dark Zurich inference sample",
        {
            "image": sample["image"].unsqueeze(0),
            "label": sample["label"].unsqueeze(0),
            "name": [sample["name"]],
            "dataset": [sample["dataset"]],
        },
    )
    assert torch.all(sample["label"] == 255)
    print("All Dark Zurich labels are ignore placeholders (255), as expected.")


def test_multiworker_loaders() -> None:
    from lambdanet_data.build import (
        build_eval_loaders,
        build_train_loader,
        load_config,
    )

    cfg = load_config("configs/data.yaml")

    print("Training loader with num_workers=2")
    train_loader = build_train_loader(
        cfg, stage="pretrain", batch_size=2, num_workers=2
    )
    check_batch("Multi-worker train batch", next(iter(train_loader)))

    print("\nEvaluation loaders with num_workers=2")
    eval_loaders = build_eval_loaders(cfg, num_workers=2)
    for name, loader in eval_loaders.items():
        check_batch(f"Multi-worker {name}", next(iter(loader)))


def test_gpu_resize_metrics() -> None:
    import torch
    import torch.nn.functional as F
    from lambdanet_data.metrics import SegMetrics

    assert torch.cuda.is_available(), "CUDA is not available to PyTorch"
    device = torch.device("cuda")

    target = torch.randint(0, 19, (1, 720, 1280), device=device)
    target[:, :20, :] = 255
    logits = torch.randn(1, 19, 128, 224, device=device)

    resized = F.interpolate(
        logits,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("Original logits:", tuple(logits.shape))
    print("Resized logits:", tuple(resized.shape))
    print("Target:", tuple(target.shape))
    assert resized.shape[-2:] == target.shape[-2:]

    metric = SegMetrics(num_classes=19, ignore_index=255)
    metric.update(resized, target)
    result = metric.compute()
    print("Random-logit integration metrics:", result)

    for key in ("mIoU", "pixel_acc", "ECE"):
        assert math.isfinite(float(result[key])), f"{key} is not finite"
    print(
        "CUDA resize and metric execution passed. "
        "Random-logit metrics are NOT model-performance results."
    )


def test_existing_smoke_test() -> None:
    run_command([sys.executable, "tests/test_smoke.py"])


def main() -> int:
    print("LAMBDANET — PHASE 1 COMPLETE VERIFICATION SUITE")
    print("Repository:", ROOT)
    print("Python executable:", sys.executable)
    print("This test suite does not train the model.")

    tests: list[tuple[str, Callable[[], None]]] = [
        ("01. Python environment and dependencies", test_environment),
        ("02. Project imports and configuration", test_project_imports_and_config),
        ("03. Dataset structure and pairing", test_dataset_structure),
        ("04. Image dimensions", test_image_dimensions),
        ("05. Label mapping", test_label_mapping),
        ("06. NightCity integrity audit", test_nightcity_integrity),
        ("07. Class distribution audit", test_class_distribution),
        ("08. Metrics unit test", test_metrics_unit),
        ("09. Real training DataLoader", test_real_train_loader),
        ("10. Real validation/test DataLoaders", test_real_eval_loaders),
        ("11. Dark Zurich inference dataset", test_dark_zurich_inference),
        ("12. Multi-worker DataLoaders", test_multiworker_loaders),
        ("13. GPU + resize + metrics integration", test_gpu_resize_metrics),
        ("14. Existing regression smoke test", test_existing_smoke_test),
    ]

    for name, fn in tests:
        record(name, fn)

    section("FINAL TEST SUMMARY")
    for name, passed, detail in RESULTS:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        if detail:
            print("       ", detail)

    passed_count = sum(1 for _, passed, _ in RESULTS if passed)
    failed_count = len(RESULTS) - passed_count
    print(f"\nTotal: {len(RESULTS)} | Passed: {passed_count} | Failed: {failed_count}")

    if failed_count:
        print("PHASE 1 VERIFICATION: FAILED — review the failed section(s).")
        return 1

    print("PHASE 1 VERIFICATION: ALL CHECKS PASSED.")
    print("Reminder: this validates the tested pipeline, not model accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
