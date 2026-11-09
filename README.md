# lambdanet_data — night-time segmentation data pipeline

Data loading, augmentation and metrics (mIoU, pixel accuracy, ECE) for the LambdaNet project.

## Setup
    pip install -r requirements.txt
    # edit configs/data.yaml: set each dataset's `root`

## Verify before training
    python tests/test_smoke.py                      # fake data, checks the code itself
    python scripts/check_data.py --samples 8        # real data: counts, label values, preview JPGs in check_out/
    python scripts/check_data.py --weights 300      # also prints class weights

If a dataset shows [FAIL] or 0 pairs, fix `img_dir`, `lbl_dir` and the suffixes in configs/data.yaml.
Release layouts differ (and Kaggle re-uploads differ again), so these are best guesses.

## Use in training
    from lambdanet_data import load_config, build_train_loader, build_eval_loaders, SegMetrics
    cfg = load_config("configs/data.yaml")
    train = build_train_loader(cfg, stage="pretrain", batch_size=4)   # NightCity
    train = build_train_loader(cfg, stage="finetune", batch_size=4)   # NightCity + BDD night x2
    evals = build_eval_loaders(cfg)                                   # dict: "dataset/split" -> loader (batch 1)

    # eval loop: label is at ORIGINAL size, so upsample logits to it before the metric
    m = SegMetrics()
    for b in evals["dark_zurich/val"]:
        logits = model(b["image"].cuda())
        logits = F.interpolate(logits, size=b["label"].shape[-2:], mode="bilinear", align_corners=False)
        m.update(logits, b["label"].cuda())
    print(m.compute())

Training batches: `image` (B,3,H,W) normalised float, `label` (B,H,W) long with 255 = ignore
(use `CrossEntropyLoss(ignore_index=255)`).

## Notes
- All label spaces are mapped to the 19 Cityscapes classes (`labels.py`).
- Dark Zurich test labels are hidden: use `dark_zurich_unlabeled` / an image-only loader to produce
  predictions for ETH's evaluation server. NDD and Dark Zurich val have very few labelled images,
  so treat their mIoU as noisy.
- BDD100K night filtering reads `timeofday` from the attribute JSON; check how many images remain.
