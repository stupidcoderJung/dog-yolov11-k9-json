#!/usr/bin/env python3
import argparse
import json
from typing import List

import torch

from dog_yolov11 import (
    DogYOLOLoss,
    DogYOLOv11,
    annotations_to_target,
    decode_dog_predictions,
    model_size_report,
)


def pick_device(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_breed_names(num_breeds: int) -> List[str]:
    names = [f"breed_{i}" for i in range(num_breeds)]
    if num_breeds > 0:
        names[0] = "Border Collie"
    if num_breeds > 1:
        names[1] = "Poodle"
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="DogYOLOv11 smoke test")
    parser.add_argument("--img-h", type=int, default=1024)
    parser.add_argument("--img-w", type=int, default=768)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--num-breeds", type=int, default=120)
    parser.add_argument("--num-emotions", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--width-mult", type=float, default=0.23)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    report = model_size_report(
        num_breeds=args.num_breeds,
        num_emotions=args.num_emotions,
        num_actions=args.num_actions,
        width_mult=args.width_mult,
    )
    print("model_size_report:", json.dumps(report, ensure_ascii=False))

    model = DogYOLOv11(
        num_breeds=args.num_breeds,
        num_emotions=args.num_emotions,
        num_actions=args.num_actions,
        width_mult=args.width_mult,
    ).to(device)
    loss_fn = DogYOLOLoss(
        num_breeds=args.num_breeds,
        num_emotions=args.num_emotions,
        num_actions=args.num_actions,
    ).to(device)

    # Minimal sample annotation (single image)
    annotations = [
        {
            "label": "Border Collie",
            "bodybndbox": [366, 750, 503, 911],
            "headbndbox": [403, 750, 462, 820],
            "emotional": "excited",
            "action": "running",
        },
        {
            "label": "Poodle",
            "bodybndbox": [662, 794, 756, 915],
            "headbndbox": [662, 811, 702, 860],
            "emotional": "curious",
            "action": "standing",
        },
        {
            "label": "Unknown",
            "bodybndbox": [177, 648, 230, 701],
            "headbndbox": [0, 0, 0, 0],
            "emotional": "calm",
            "action": "resting",
        },
    ]

    breed_to_idx = {"Border Collie": 0, "Poodle": 1}
    emotion_to_idx = {"excited": 0, "curious": 1, "calm": 2, "resting": 3}
    action_to_idx = {"running": 0, "standing": 1, "resting": 2}

    target = annotations_to_target(
        annotations,
        breed_to_idx=breed_to_idx,
        emotion_to_idx=emotion_to_idx,
        action_to_idx=action_to_idx,
        unknown_breed_policy="ignore",
    )
    targets = [target for _ in range(args.batch)]

    x = torch.randn(args.batch, 3, args.img_h, args.img_w, device=device)

    model.train()
    preds = model(x)
    print("pred_shapes:", [tuple(p.shape) for p in preds])

    loss = loss_fn(preds, targets, img_size=(args.img_h, args.img_w))
    print("loss:", float(loss.detach().cpu().item()))

    loss.backward()
    print("backward: ok")

    model.eval()
    with torch.no_grad():
        preds_eval = model(x)
        breed_names = build_breed_names(args.num_breeds)
        emotion_names = ["excited", "curious", "calm", "resting", "other"][: args.num_emotions]
        while len(emotion_names) < args.num_emotions:
            emotion_names.append(f"emotion_{len(emotion_names)}")
        action_names = ["running", "standing", "resting", "walking", "playing"][: args.num_actions]
        while len(action_names) < args.num_actions:
            action_names.append(f"action_{len(action_names)}")

        decoded = decode_dog_predictions(
            preds_eval,
            image_size=(args.img_h, args.img_w),
            breed_names=breed_names,
            emotion_names=emotion_names,
            action_names=action_names,
            obj_thres=0.05,
            conf_thres=0.25,
            iou_thres=0.50,
            apply_nms=True,
            class_agnostic=False,
            max_det=20,
        )
    print("decoded_count_image0:", len(decoded[0]) if len(decoded) > 0 else 0)
    if len(decoded) > 0 and len(decoded[0]) > 0:
        print("decoded_sample_image0:", json.dumps(decoded[0][0], ensure_ascii=False))


if __name__ == "__main__":
    main()
