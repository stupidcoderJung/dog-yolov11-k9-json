#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dog_yolov11 import DogYOLOv11
from experiments.issue11_roi_v2.roi_attr_head import DogRoiAttrHead
from experiments.issue11_roi_v2.roi_v2_adapter import RoiV2HybridExperimentModel


def pick_device(user_device: str) -> str:
    if user_device != "auto":
        return user_device
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue-11 ROI v2 inference demo")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--img-h", type=int, default=640)
    parser.add_argument("--img-w", type=int, default=640)
    parser.add_argument("--num-breeds", type=int, default=12)
    parser.add_argument("--num-emotions", type=int, default=5)
    parser.add_argument("--num-actions", type=int, default=5)
    parser.add_argument("--width-mult", type=float, default=0.23)
    parser.add_argument("--roi-fusion", type=str, default="concat", choices=["concat", "xattn"])
    parser.add_argument("--roi-use-multiscale", action="store_true")
    parser.add_argument("--obj-thres", type=float, default=0.01)
    parser.add_argument("--conf-thres", type=float, default=0.01)
    parser.add_argument(
        "--score-policy",
        type=str,
        default="obj_x_breed",
        choices=["obj_x_breed", "calibrated_obj_x_breed", "breed_only"],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    args = parser.parse_args()

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
        num_breeds=args.num_breeds,
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
    model.eval()

    images = torch.randn(args.batch, 3, args.img_h, args.img_w, device=device)

    decoded = model.infer_with_roi_attributes(
        images,
        breed_names=breed_names,
        emotion_names=emotion_names,
        action_names=action_names,
        obj_thres=args.obj_thres,
        conf_thres=args.conf_thres,
    )

    counts = [len(x) for x in decoded]
    print("decoded_counts:", counts)
    if decoded and decoded[0]:
        sample = decoded[0][0]
        print("sample_keys:", sorted(sample.keys()))
        print(
            "sample_fields:",
            {
                "label": sample.get("label"),
                "emotional": sample.get("emotional"),
                "action": sample.get("action"),
                "objectness": sample.get("objectness"),
                "breed_confidence": sample.get("breed_confidence"),
                "final_confidence": sample.get("final_confidence"),
                "confidence": sample.get("confidence"),
                "score_policy": sample.get("score_policy"),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
