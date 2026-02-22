#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg

from dataset import (
    DogJsonDataset,
    collect_unique_values,
    dog_collate_fn,
    load_class_names,
)
from dog_yolov11 import DogYOLOLoss, DogYOLOv11, annotations_to_target, model_size_report
from label_validator import validate_manifest


DEFAULT_EMOTIONS = ["excited", "curious", "calm", "resting", "other"]
DEFAULT_ACTIONS = ["running", "standing", "resting", "walking", "playing"]


def pick_device(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_all(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _xyxy_to_coco(box: List[int]) -> List[int]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def create_synthetic_manifest(
    out_dir: Path,
    num_samples: int,
    image_size: Tuple[int, int],
    breed_names: List[str],
    emotion_names: List[str],
    action_names: List[str],
    seed: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    h, w = image_size
    records: List[Dict[str, Any]] = []

    for i in range(num_samples):
        img = torch.randint(0, 256, (3, h, w), dtype=torch.uint8)
        image_name = f"sample_{i:04d}.jpg"
        image_path = images_dir / image_name
        write_jpeg(img, str(image_path), quality=95)

        ann_count = rng.randint(1, 3)
        anns: List[Dict[str, Any]] = []
        for _ in range(ann_count):
            bw = rng.randint(max(24, w // 12), max(32, w // 4))
            bh = rng.randint(max(24, h // 12), max(32, h // 4))
            x1 = rng.randint(0, max(0, w - bw - 1))
            y1 = rng.randint(0, max(0, h - bh - 1))
            x2 = min(w - 1, x1 + bw)
            y2 = min(h - 1, y1 + bh)

            has_head = rng.random() > 0.2
            if has_head:
                hbw = max(8, int((x2 - x1) * rng.uniform(0.25, 0.5)))
                hbh = max(8, int((y2 - y1) * rng.uniform(0.2, 0.45)))
                hx1 = rng.randint(x1, max(x1, x2 - hbw))
                hy1 = rng.randint(y1, max(y1, y2 - hbh))
                hx2 = min(x2, hx1 + hbw)
                hy2 = min(y2, hy1 + hbh)
                head_xyxy = [hx1, hy1, hx2, hy2]
            else:
                head_xyxy = [0, 0, 0, 0]

            body_xyxy = [x1, y1, x2, y2]
            breed = rng.choice(breed_names) if breed_names else "Unknown"
            emo = rng.choice(emotion_names) if emotion_names else "calm"
            act = rng.choice(action_names) if action_names else "standing"

            anns.append(
                {
                    "label": breed,
                    "bodybndbox": body_xyxy,
                    "bodybndbox_coco": _xyxy_to_coco(body_xyxy),
                    "headbndbox": head_xyxy,
                    "headbndbox_coco": _xyxy_to_coco(head_xyxy),
                    "emotional": emo,
                    "action": act,
                }
            )

        records.append({"image": f"images/{image_name}", "annotations": anns})

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest_path


def build_class_names(
    dataset: DogJsonDataset,
    breed_file: str | None,
    emotion_file: str | None,
    action_file: str | None,
    unknown_breed_policy: str,
) -> Tuple[List[str], List[str], List[str]]:
    breed_names = load_class_names(breed_file)
    emotion_names = load_class_names(emotion_file)
    action_names = load_class_names(action_file)

    if not breed_names:
        breed_names = [x for x in collect_unique_values(dataset.samples, "label") if x != "Unknown"]

    if not emotion_names:
        emotion_names = collect_unique_values(dataset.samples, "emotional")
        if not emotion_names:
            emotion_names = DEFAULT_EMOTIONS.copy()

    if not action_names:
        action_names = collect_unique_values(dataset.samples, "action")
        if not action_names:
            action_names = DEFAULT_ACTIONS.copy()

    if unknown_breed_policy == "class" and "Unknown" not in breed_names:
        breed_names.append("Unknown")

    if len(breed_names) == 0:
        raise ValueError("No breed classes found. Provide --breed-names or valid labels in manifest")

    return breed_names, emotion_names, action_names


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def assert_resume_class_compatibility(
    ckpt: Dict[str, Any],
    *,
    breed_names: List[str],
    emotion_names: List[str],
    action_names: List[str],
) -> None:
    mismatches: List[str] = []
    details: List[str] = []
    for field, current in (
        ("breed_names", breed_names),
        ("emotion_names", emotion_names),
        ("action_names", action_names),
    ):
        saved = ckpt.get(field)
        if saved is None:
            continue
        saved_list = list(saved)
        current_list = list(current)
        if saved_list != current_list:
            mismatches.append(field)
            details.append(f"{field}: checkpoint={saved_list} current={current_list}")

    if mismatches:
        mismatch_fields = ", ".join(mismatches)
        mismatch_details = "; ".join(details)
        raise ValueError(
            f"Resume blocked due to class-order mismatch ({mismatch_fields}). {mismatch_details}"
        )


def compute_step_lr(
    step: int,
    *,
    base_lr: float,
    min_lr: float,
    total_steps: int,
    warmup_steps: int,
    schedule: str,
) -> float:
    if schedule == "none":
        return float(base_lr)

    s = max(int(step), 0)
    total = max(int(total_steps), 1)
    warmup = max(int(warmup_steps), 0)

    if warmup > 0 and s < warmup:
        return float(base_lr) * float(s + 1) / float(max(warmup, 1))

    decay_start = warmup
    decay_end = total - 1
    if decay_end <= decay_start:
        return float(min_lr)

    progress = float(s - decay_start) / float(decay_end - decay_start)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(min_lr) + (float(base_lr) - float(min_lr)) * cosine


def main() -> int:
    parser = argparse.ArgumentParser(description="DogYOLOv11 training script")
    parser.add_argument("--manifest", type=str, default=None, help="Path to dataset manifest json")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-h", type=int, default=640)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr-schedule", type=str, default="none", choices=["none", "cosine"])
    parser.add_argument("--warmup-epochs", type=float, default=0.0)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--width-mult", type=float, default=0.23)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0)
    parser.add_argument("--unknown-breed-policy", type=str, default="ignore", choices=["ignore", "class"])
    parser.add_argument("--breed-names", type=str, default=None)
    parser.add_argument("--emotion-names", type=str, default=None)
    parser.add_argument("--action-names", type=str, default=None)
    parser.add_argument("--validate-labels", action="store_true")
    parser.add_argument("--allow-unknown-breed", action="store_true")
    parser.add_argument("--save-dir", type=str, default="runs/train")
    parser.add_argument("--run-name", type=str, default="exp")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint (last.pt) to resume")
    parser.add_argument("--synthetic-samples", type=int, default=0, help="If >0, auto-generate synthetic dataset")
    args = parser.parse_args()

    seed_all(args.seed)
    device = pick_device(args.device)

    resume_path: Path | None = None
    if args.resume_from:
        resume_path = Path(args.resume_from).resolve()
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        run_dir = resume_path.parent
    else:
        run_stamp = time.strftime("%Y%m%d-%H%M%S")
        run_dir = Path(args.save_dir) / f"{args.run_name}-{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.manifest
    if args.synthetic_samples > 0:
        synthetic_breeds = load_class_names(args.breed_names) or [
            "Border Collie",
            "Poodle",
            "Golden Retriever",
        ]
        synthetic_emotions = load_class_names(args.emotion_names) or DEFAULT_EMOTIONS.copy()
        synthetic_actions = load_class_names(args.action_names) or DEFAULT_ACTIONS.copy()

        manifest_path = str(
            create_synthetic_manifest(
                out_dir=run_dir / "synthetic_dataset",
                num_samples=args.synthetic_samples,
                image_size=(args.img_h, args.img_w),
                breed_names=synthetic_breeds,
                emotion_names=synthetic_emotions,
                action_names=synthetic_actions,
                seed=args.seed,
            )
        )
        print(f"synthetic_manifest: {manifest_path}")

    if not manifest_path:
        raise ValueError("--manifest is required when --synthetic-samples is 0")

    dataset = DogJsonDataset(
        manifest_path=manifest_path,
        image_size=(args.img_h, args.img_w),
        normalize=True,
        strict_image_exists=True,
    )

    breed_names, emotion_names, action_names = build_class_names(
        dataset,
        breed_file=args.breed_names,
        emotion_file=args.emotion_names,
        action_file=args.action_names,
        unknown_breed_policy=args.unknown_breed_policy,
    )

    if args.validate_labels:
        report = validate_manifest(
            manifest_path=manifest_path,
            breed_names=breed_names,
            emotion_names=emotion_names,
            action_names=action_names,
            allow_unknown_breed=args.allow_unknown_breed,
            max_messages=200,
        )
        print("label_validation:", json.dumps(report, ensure_ascii=False))
        if not report["is_valid"]:
            raise ValueError("Label validation failed. Fix dataset first.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=dog_collate_fn,
        drop_last=False,
    )

    steps_per_epoch = len(loader)
    if args.max_steps_per_epoch > 0:
        steps_per_epoch = min(steps_per_epoch, int(args.max_steps_per_epoch))
    steps_per_epoch = max(int(steps_per_epoch), 1)
    total_train_steps = max(int(args.epochs) * steps_per_epoch, 1)
    warmup_steps = int(max(float(args.warmup_epochs), 0.0) * steps_per_epoch)
    min_lr = min(float(args.min_lr), float(args.lr))

    breed_to_idx = {name: i for i, name in enumerate(breed_names)}
    emotion_to_idx = {name: i for i, name in enumerate(emotion_names)}
    action_to_idx = {name: i for i, name in enumerate(action_names)}

    model = DogYOLOv11(
        num_breeds=len(breed_names),
        num_emotions=len(emotion_names),
        num_actions=len(action_names),
        width_mult=args.width_mult,
    ).to(device)

    loss_fn = DogYOLOLoss(
        num_breeds=len(breed_names),
        num_emotions=len(emotion_names),
        num_actions=len(action_names),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    global_step = 0
    start_epoch = 1
    if resume_path is not None:
        ckpt = torch.load(str(resume_path), map_location=device)
        if isinstance(ckpt, dict):
            assert_resume_class_compatibility(
                ckpt,
                breed_names=breed_names,
                emotion_names=emotion_names,
                action_names=action_names,
            )
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        if isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if isinstance(ckpt, dict):
            global_step = int(ckpt.get("global_step", 0))
            start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(
            f"resume_from: {resume_path} start_epoch={start_epoch} "
            f"target_epochs={args.epochs} global_step={global_step}"
        )

    size_info = model_size_report(
        num_breeds=len(breed_names),
        num_emotions=len(emotion_names),
        num_actions=len(action_names),
        width_mult=args.width_mult,
    )
    print("model_size_report:", json.dumps(size_info, ensure_ascii=False))

    save_json(
        run_dir / "class_names.json",
        {
            "breed_names": breed_names,
            "emotion_names": emotion_names,
            "action_names": action_names,
        },
    )

    save_json(
        run_dir / "train_config.json",
        {
            "manifest": str(Path(manifest_path).resolve()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "img_h": args.img_h,
            "img_w": args.img_w,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_schedule": args.lr_schedule,
            "warmup_epochs": args.warmup_epochs,
            "min_lr": min_lr,
            "steps_per_epoch": steps_per_epoch,
            "total_train_steps": total_train_steps,
            "warmup_steps": warmup_steps,
            "width_mult": args.width_mult,
            "unknown_breed_policy": args.unknown_breed_policy,
            "seed": args.seed,
            "device": device,
            "resume_from": str(resume_path) if resume_path is not None else None,
            "start_epoch": int(start_epoch),
            "initial_global_step": int(global_step),
        },
    )

    if start_epoch > args.epochs:
        print(f"nothing_to_do: start_epoch={start_epoch} > epochs={args.epochs}")
        return 0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        step_count = 0

        for step_idx, (images, ann_lists, _meta) in enumerate(loader, start=1):
            current_lr = compute_step_lr(
                global_step,
                base_lr=float(args.lr),
                min_lr=min_lr,
                total_steps=total_train_steps,
                warmup_steps=warmup_steps,
                schedule=args.lr_schedule,
            )
            for pg in optimizer.param_groups:
                pg["lr"] = current_lr

            images = images.to(device)
            targets = [
                annotations_to_target(
                    anns,
                    breed_to_idx=breed_to_idx,
                    emotion_to_idx=emotion_to_idx,
                    action_to_idx=action_to_idx,
                    unknown_breed_policy=args.unknown_breed_policy,
                )
                for anns in ann_lists
            ]

            preds = model(images)
            loss = loss_fn(preds, targets, img_size=(args.img_h, args.img_w))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            epoch_loss += loss_val
            step_count += 1
            global_step += 1

            print(
                f"epoch={epoch}/{args.epochs} step={step_idx}/{len(loader)} "
                f"global_step={global_step} loss={loss_val:.6f} lr={current_lr:.8f}"
            )

            if args.max_steps_per_epoch > 0 and step_idx >= args.max_steps_per_epoch:
                break

        mean_loss = epoch_loss / max(step_count, 1)
        print(f"epoch={epoch} mean_loss={mean_loss:.6f}")

        ckpt = {
            "epoch": epoch,
            "global_step": global_step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "breed_names": breed_names,
            "emotion_names": emotion_names,
            "action_names": action_names,
            "args": vars(args),
        }
        torch.save(ckpt, run_dir / "last.pt")

    print(f"train_done: run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
