#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict, List

import torch

ROOT = Path(__file__).resolve().parents[2]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dog_yolov11 import DogYOLOLoss, DogYOLOv11, annotations_to_target
from experiments.issue8_roi_attr.roi_attr_head import DogRoiAttrHead
from experiments.issue8_roi_attr.roi_attr_loss import RoiAttributeLoss
from experiments.issue11_roi_v2.roi_v2_adapter import RoiV2HybridExperimentModel


def pick_device(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def random_annotations(
    *,
    rng: random.Random,
    image_h: int,
    image_w: int,
    count: int,
    breed_names: List[str],
    emotion_names: List[str],
    action_names: List[str],
) -> List[Dict[str, Any]]:
    anns: List[Dict[str, Any]] = []
    for _ in range(count):
        bw = rng.randint(max(20, image_w // 16), max(30, image_w // 4))
        bh = rng.randint(max(20, image_h // 16), max(30, image_h // 4))
        x1 = rng.randint(0, max(0, image_w - bw - 1))
        y1 = rng.randint(0, max(0, image_h - bh - 1))
        x2 = min(image_w - 1, x1 + bw)
        y2 = min(image_h - 1, y1 + bh)

        has_head = rng.random() > 0.25
        if has_head:
            hw = max(8, int((x2 - x1) * rng.uniform(0.2, 0.6)))
            hh = max(8, int((y2 - y1) * rng.uniform(0.2, 0.6)))
            hx1 = rng.randint(x1, max(x1, x2 - hw))
            hy1 = rng.randint(y1, max(y1, y2 - hh))
            hx2 = min(x2, hx1 + hw)
            hy2 = min(y2, hy1 + hh)
            head = [hx1, hy1, hx2, hy2]
        else:
            head = [0, 0, 0, 0]

        anns.append(
            {
                "label": rng.choice(breed_names),
                "bodybndbox": [x1, y1, x2, y2],
                "headbndbox": head,
                "emotional": rng.choice(emotion_names),
                "action": rng.choice(action_names),
            }
        )
    return anns


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue-11 ROI v2 smoke test")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--img-h", type=int, default=640)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--width-mult", type=float, default=0.23)
    parser.add_argument("--num-breeds", type=int, default=12)
    parser.add_argument("--num-emotions", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--roi-output-size", type=int, default=7)
    parser.add_argument("--roi-hidden-dim", type=int, default=192)
    parser.add_argument("--roi-fusion", type=str, default="concat", choices=["concat", "xattn"])
    parser.add_argument("--roi-use-multiscale", action="store_true")
    parser.add_argument("--with-breed-head", action="store_true", default=True)
    parser.add_argument("--lambda-attr-grid", type=float, default=0.35)
    parser.add_argument("--lambda-attr-roi", type=float, default=1.0)
    parser.add_argument(
        "--score-policy",
        type=str,
        default="obj_x_breed",
        choices=["obj_x_breed", "calibrated_obj_x_breed", "breed_only"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)

    breed_names = [f"breed_{i}" for i in range(args.num_breeds)]
    emotion_names = [f"emotion_{i}" for i in range(args.num_emotions)]
    action_names = [f"action_{i}" for i in range(args.num_actions)]

    detector = DogYOLOv11(
        num_breeds=args.num_breeds,
        num_emotions=args.num_emotions,
        num_actions=args.num_actions,
        width_mult=args.width_mult,
    ).to(device)

    roi_head = DogRoiAttrHead(
        in_channels=detector.backbone.out_channels,
        num_emotions=args.num_emotions,
        num_actions=args.num_actions,
        num_breeds=args.num_breeds if args.with_breed_head else None,
        roi_output_size=args.roi_output_size,
        hidden_dim=args.roi_hidden_dim,
        fusion=args.roi_fusion,
        use_multiscale=args.roi_use_multiscale,
        feature_strides=detector.strides,
    ).to(device)

    model = RoiV2HybridExperimentModel(
        detector=detector,
        roi_head=roi_head,
        score_policy=args.score_policy,
        calibrated_temperature=args.temperature,
    ).to(device)

    det_loss_fn = DogYOLOLoss(
        num_breeds=args.num_breeds,
        num_emotions=args.num_emotions,
        num_actions=args.num_actions,
        lambda_attr=args.lambda_attr_grid,
    ).to(device)
    roi_loss_fn = RoiAttributeLoss(
        lambda_attr_roi=args.lambda_attr_roi,
        with_breed_head=args.with_breed_head,
    ).to(device)

    images = torch.randn(args.batch, 3, args.img_h, args.img_w, device=device)

    targets: List[Dict[str, torch.Tensor]] = []
    breed_to_idx = {v: i for i, v in enumerate(breed_names)}
    emotion_to_idx = {v: i for i, v in enumerate(emotion_names)}
    action_to_idx = {v: i for i, v in enumerate(action_names)}

    for _ in range(args.batch):
        anns = random_annotations(
            rng=rng,
            image_h=args.img_h,
            image_w=args.img_w,
            count=rng.randint(1, 4),
            breed_names=breed_names,
            emotion_names=emotion_names,
            action_names=action_names,
        )
        targets.append(
            annotations_to_target(
                anns,
                breed_to_idx=breed_to_idx,
                emotion_to_idx=emotion_to_idx,
                action_to_idx=action_to_idx,
                unknown_breed_policy="ignore",
            )
        )

    body_boxes = [t["body_boxes"].to(device) for t in targets]
    head_boxes = [t["head_boxes"].to(device) for t in targets]
    head_valid = [t["head_valid"].to(device) for t in targets]

    model.train()
    out = model(images, body_boxes=body_boxes, head_boxes=head_boxes, head_valid=head_valid)

    det_loss = det_loss_fn(out["preds"], targets, img_size=(args.img_h, args.img_w))
    roi_losses = roi_loss_fn(out["roi"], targets)
    total_loss = det_loss + roi_losses["loss"]
    total_loss.backward()

    model.eval()
    with torch.no_grad():
        decoded = model.infer_with_roi_attributes(
            images[:1],
            breed_names=breed_names,
            emotion_names=emotion_names,
            action_names=action_names,
            obj_thres=0.01,
            conf_thres=0.01,
            max_det=20,
        )

    sample_scores = None
    if decoded and decoded[0]:
        s = decoded[0][0]
        sample_scores = {
            "objectness": s.get("objectness"),
            "breed_confidence": s.get("breed_confidence"),
            "final_confidence": s.get("final_confidence"),
            "score_policy": s.get("score_policy"),
        }

    print(
        "smoke_roi_v2:",
        {
            "batch": args.batch,
            "fusion": args.roi_fusion,
            "multiscale": args.roi_use_multiscale,
            "with_breed_head": args.with_breed_head,
            "lambda_attr_grid": args.lambda_attr_grid,
            "lambda_attr_roi": args.lambda_attr_roi,
            "score_policy": args.score_policy,
            "temperature": args.temperature,
            "det_loss": float(det_loss.detach().cpu().item()),
            "roi_loss": float(roi_losses["loss"].detach().cpu().item()),
            "total_loss": float(total_loss.detach().cpu().item()),
            "roi_samples": int(out["roi"]["batch_indices"].shape[0]),
            "decoded_counts": [len(x) for x in decoded],
            "sample_scores": sample_scores,
        },
    )
    print("backward: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
