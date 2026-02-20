#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

from dataset import DogJsonDataset, collect_unique_values, dog_collate_fn, load_class_names
from dog_yolov11 import DogYOLOLoss, DogYOLOv11, annotations_to_target, decode_dog_predictions


def pick_device(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_ckpt_if_exists(model: DogYOLOv11, checkpoint: str | None, device: str) -> Dict[str, Any]:
    if not checkpoint:
        return {}
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location=device)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=True)
    return ckpt if isinstance(ckpt, dict) else {}


def infer_class_names(
    dataset: DogJsonDataset,
    checkpoint_dict: Dict[str, Any],
    breed_file: str | None,
    emotion_file: str | None,
    action_file: str | None,
    unknown_breed_policy: str,
) -> Tuple[List[str], List[str], List[str]]:
    breed_names = load_class_names(breed_file)
    emotion_names = load_class_names(emotion_file)
    action_names = load_class_names(action_file)
    breed_from_checkpoint = False

    if not breed_names and checkpoint_dict.get("breed_names"):
        breed_names = list(checkpoint_dict["breed_names"])
        breed_from_checkpoint = True
    if not emotion_names and checkpoint_dict.get("emotion_names"):
        emotion_names = list(checkpoint_dict["emotion_names"])
    if not action_names and checkpoint_dict.get("action_names"):
        action_names = list(checkpoint_dict["action_names"])

    if not breed_names:
        breed_names = [x for x in collect_unique_values(dataset.samples, "label") if x != "Unknown"]
    if not emotion_names:
        emotion_names = collect_unique_values(dataset.samples, "emotional")
    if not action_names:
        action_names = collect_unique_values(dataset.samples, "action")

    # Keep checkpoint class layout unchanged to avoid head shape mismatch on strict load.
    if (
        unknown_breed_policy == "class"
        and "Unknown" not in breed_names
        and not breed_from_checkpoint
    ):
        breed_names.append("Unknown")

    if len(breed_names) == 0:
        raise ValueError("Could not infer breed class names")
    if len(emotion_names) == 0:
        raise ValueError("Could not infer emotion class names")
    if len(action_names) == 0:
        raise ValueError("Could not infer action class names")

    return breed_names, emotion_names, action_names


def _xyxy_to_coco(box: List[int]) -> List[int]:
    x1, y1, x2, y2 = box
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def _scale_xyxy_to_orig(
    box: Any,
    resized_size: Tuple[int, int],
    orig_size: Tuple[int, int],
) -> List[int]:
    if not isinstance(box, list) or len(box) != 4:
        return [0, 0, 0, 0]

    x1, y1, x2, y2 = [float(v) for v in box]
    if x2 <= x1 or y2 <= y1:
        return [0, 0, 0, 0]

    resized_h, resized_w = resized_size
    orig_h, orig_w = orig_size
    sx = float(orig_w) / max(float(resized_w), 1.0)
    sy = float(orig_h) / max(float(resized_h), 1.0)

    ox1 = int(round(x1 * sx))
    oy1 = int(round(y1 * sy))
    ox2 = int(round(x2 * sx))
    oy2 = int(round(y2 * sy))

    ox1 = min(max(ox1, 0), max(orig_w - 1, 0))
    oy1 = min(max(oy1, 0), max(orig_h - 1, 0))
    ox2 = min(max(ox2, 0), max(orig_w - 1, 0))
    oy2 = min(max(oy2, 0), max(orig_h - 1, 0))

    if ox2 <= ox1 or oy2 <= oy1:
        return [0, 0, 0, 0]
    return [ox1, oy1, ox2, oy2]


def _remap_predictions_to_original(
    predictions: List[Dict[str, Any]],
    resized_size: Tuple[int, int],
    orig_size: Tuple[int, int],
) -> List[Dict[str, Any]]:
    remapped: List[Dict[str, Any]] = []
    for rec in predictions:
        out = dict(rec)

        body = _scale_xyxy_to_orig(out.get("bodybndbox", [0, 0, 0, 0]), resized_size, orig_size)
        head = _scale_xyxy_to_orig(out.get("headbndbox", [0, 0, 0, 0]), resized_size, orig_size)

        out["bodybndbox"] = body
        out["bodybndbox_coco"] = _xyxy_to_coco(body)
        out["headbndbox"] = head
        out["headbndbox_coco"] = _xyxy_to_coco(head)
        remapped.append(out)
    return remapped


def main() -> int:
    parser = argparse.ArgumentParser(description="DogYOLOv11 validation/eval script")
    parser.add_argument("--manifest", required=True, type=str)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--img-h", type=int, default=640)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--width-mult", type=float, default=0.23)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--unknown-breed-policy", type=str, default="ignore", choices=["ignore", "class"])
    parser.add_argument("--breed-names", type=str, default=None)
    parser.add_argument("--emotion-names", type=str, default=None)
    parser.add_argument("--action-names", type=str, default=None)
    parser.add_argument("--obj-thres", type=float, default=0.05)
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--iou-thres", type=float, default=0.50)
    parser.add_argument("--class-agnostic", action="store_true")
    parser.add_argument("--max-det", type=int, default=100)
    parser.add_argument("--output-json", type=str, default="runs/val/predictions.json")
    parser.add_argument("--max-batches", type=int, default=0)
    args = parser.parse_args()

    device = pick_device(args.device)

    dataset = DogJsonDataset(
        manifest_path=args.manifest,
        image_size=(args.img_h, args.img_w),
        normalize=True,
        strict_image_exists=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=dog_collate_fn,
        drop_last=False,
    )

    # Build model shape from checkpoint names if available.
    ckpt_preview: Dict[str, Any] = {}
    if args.checkpoint:
        ckpt_preview = torch.load(args.checkpoint, map_location="cpu")
        if not isinstance(ckpt_preview, dict):
            ckpt_preview = {}

    breed_names, emotion_names, action_names = infer_class_names(
        dataset=dataset,
        checkpoint_dict=ckpt_preview,
        breed_file=args.breed_names,
        emotion_file=args.emotion_names,
        action_file=args.action_names,
        unknown_breed_policy=args.unknown_breed_policy,
    )

    model = DogYOLOv11(
        num_breeds=len(breed_names),
        num_emotions=len(emotion_names),
        num_actions=len(action_names),
        width_mult=args.width_mult,
    ).to(device)
    ckpt = load_ckpt_if_exists(model, args.checkpoint, device)

    loss_fn = DogYOLOLoss(
        num_breeds=len(breed_names),
        num_emotions=len(emotion_names),
        num_actions=len(action_names),
    ).to(device)

    breed_to_idx = {name: i for i, name in enumerate(breed_names)}
    emotion_to_idx = {name: i for i, name in enumerate(emotion_names)}
    action_to_idx = {name: i for i, name in enumerate(action_names)}

    model.eval()
    total_loss = 0.0
    total_batches = 0
    nms_before = 0
    nms_after = 0

    outputs: List[Dict[str, Any]] = []

    with torch.no_grad():
        for bidx, (images, ann_lists, meta) in enumerate(loader, start=1):
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
            total_loss += float(loss.detach().cpu().item())
            total_batches += 1

            # For pre-NMS counting, avoid max_det truncation by using
            # the theoretical maximum per-image candidates across scales.
            max_det_pre_nms = int(sum(int(p.shape[1]) * int(p.shape[2]) for p in preds))
            decoded_no_nms = decode_dog_predictions(
                preds,
                image_size=(args.img_h, args.img_w),
                breed_names=breed_names,
                emotion_names=emotion_names,
                action_names=action_names,
                obj_thres=args.obj_thres,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                apply_nms=False,
                class_agnostic=args.class_agnostic,
                max_det=max_det_pre_nms,
            )
            decoded_nms = decode_dog_predictions(
                preds,
                image_size=(args.img_h, args.img_w),
                breed_names=breed_names,
                emotion_names=emotion_names,
                action_names=action_names,
                obj_thres=args.obj_thres,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                apply_nms=True,
                class_agnostic=args.class_agnostic,
                max_det=args.max_det,
            )

            for i in range(len(decoded_nms)):
                before_cnt = len(decoded_no_nms[i])
                after_cnt = len(decoded_nms[i])
                nms_before += before_cnt
                nms_after += after_cnt

                orig_h, orig_w = meta["orig_sizes"][i]
                resized_h, resized_w = meta["image_sizes"][i]
                preds_orig = _remap_predictions_to_original(
                    decoded_nms[i],
                    resized_size=(int(resized_h), int(resized_w)),
                    orig_size=(int(orig_h), int(orig_w)),
                )

                outputs.append(
                    {
                        "image_path": meta["image_paths"][i],
                        "prediction_space": "original_image",
                        "orig_size": [int(orig_h), int(orig_w)],
                        "resized_size": [int(resized_h), int(resized_w)],
                        "predictions": preds_orig,
                        "nms_before_count": before_cnt,
                        "nms_after_count": after_cnt,
                    }
                )

            if args.max_batches > 0 and bidx >= args.max_batches:
                break

    mean_loss = total_loss / max(total_batches, 1)
    summary = {
        "manifest": str(Path(args.manifest).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()) if args.checkpoint else None,
        "ckpt_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "num_images": len(outputs),
        "mean_loss": mean_loss,
        "nms_before_total": nms_before,
        "nms_after_total": nms_after,
        "nms_reduction": nms_before - nms_after,
        "obj_thres": args.obj_thres,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "class_agnostic": args.class_agnostic,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "results": outputs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved_predictions: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
