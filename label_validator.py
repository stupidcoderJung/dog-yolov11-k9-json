#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from torchvision.io import ImageReadMode, read_image

from dataset import load_class_names, load_manifest


def _is_box_like(box: Any) -> bool:
    return isinstance(box, list) and len(box) == 4


def _as_float_box(box: Sequence[Any]) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return float(x1), float(y1), float(x2), float(y2)


def _try_as_float_box(box: Sequence[Any]) -> Optional[Tuple[float, float, float, float]]:
    try:
        return _as_float_box(box)
    except (TypeError, ValueError):
        return None


def _box_errors(box: Sequence[Any], width: int, height: int, name: str) -> List[str]:
    errs: List[str] = []
    if not _is_box_like(box):
        return [f"{name}: expected [x1,y1,x2,y2]"]

    parsed = _try_as_float_box(box)
    if parsed is None:
        return [f"{name}: non-numeric coordinate value"]
    x1, y1, x2, y2 = parsed

    if x2 <= x1 or y2 <= y1:
        errs.append(f"{name}: invalid area (x2<=x1 or y2<=y1)")

    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        errs.append(f"{name}: out-of-image range for image size ({width},{height})")
    return errs


def validate_annotation(
    ann: Dict[str, Any],
    width: int,
    height: int,
    breed_set: Optional[set] = None,
    emotion_set: Optional[set] = None,
    action_set: Optional[set] = None,
    allow_unknown_breed: bool = True,
) -> Tuple[List[str], List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    for req in ["label", "bodybndbox", "emotional", "action"]:
        if req not in ann:
            errors.append(f"missing required key: {req}")

    label = str(ann.get("label", ""))
    emo = str(ann.get("emotional", ""))
    action = str(ann.get("action", ""))

    body = ann.get("bodybndbox", None)
    if body is None:
        errors.append("missing bodybndbox")
    else:
        errors.extend(_box_errors(body, width, height, "bodybndbox"))

    head = ann.get("headbndbox", [0, 0, 0, 0])
    if not _is_box_like(head):
        errors.append("headbndbox: expected [x1,y1,x2,y2]")
    else:
        head_parsed = _try_as_float_box(head)
        if head_parsed is None:
            errors.append("headbndbox: non-numeric coordinate value")
            head_parsed = (0.0, 0.0, 0.0, 0.0)
        hx1, hy1, hx2, hy2 = head_parsed
        has_head = (hx2 > hx1) and (hy2 > hy1)
        if has_head:
            errors.extend(_box_errors(head, width, height, "headbndbox"))
            if _is_box_like(body):
                body_parsed = _try_as_float_box(body)
                if body_parsed is not None:
                    bx1, by1, bx2, by2 = body_parsed
                    if hx1 < bx1 or hy1 < by1 or hx2 > bx2 or hy2 > by2:
                        warnings.append("headbndbox is not fully inside bodybndbox")

    if breed_set is not None and label not in breed_set:
        if not (allow_unknown_breed and label == "Unknown"):
            errors.append(f"unknown breed label: {label}")

    if emotion_set is not None and emo not in emotion_set:
        errors.append(f"unknown emotional label: {emo}")

    if action_set is not None and action not in action_set:
        errors.append(f"unknown action label: {action}")

    return errors, warnings


def validate_manifest(
    manifest_path: str,
    breed_names: Optional[List[str]] = None,
    emotion_names: Optional[List[str]] = None,
    action_names: Optional[List[str]] = None,
    allow_unknown_breed: bool = True,
    max_messages: int = 100,
) -> Dict[str, Any]:
    samples = load_manifest(manifest_path)
    manifest_dir = Path(manifest_path).resolve().parent

    breed_set = set(breed_names) if breed_names else None
    emotion_set = set(emotion_names) if emotion_names else None
    action_set = set(action_names) if action_names else None

    total_samples = 0
    total_annotations = 0
    error_count = 0
    warning_count = 0

    messages: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples):
        total_samples += 1

        image_rel = (
            sample.get("image")
            or sample.get("image_path")
            or sample.get("image_file")
            or sample.get("file_name")
        )
        if image_rel is None:
            error_count += 1
            if len(messages) < max_messages:
                messages.append({"sample": i, "type": "error", "msg": "missing image path"})
            continue

        image_path = Path(str(image_rel))
        if not image_path.is_absolute():
            image_path = (manifest_dir / image_path).resolve()

        if not image_path.exists():
            error_count += 1
            if len(messages) < max_messages:
                messages.append({
                    "sample": i,
                    "image": str(image_path),
                    "type": "error",
                    "msg": "image file does not exist",
                })
            continue

        try:
            img = read_image(str(image_path), mode=ImageReadMode.RGB)
            _, h, w = img.shape
        except Exception as e:
            error_count += 1
            if len(messages) < max_messages:
                messages.append({
                    "sample": i,
                    "image": str(image_path),
                    "type": "error",
                    "msg": f"failed to read image: {e}",
                })
            continue

        anns = sample.get("annotations", [])
        if not isinstance(anns, list):
            error_count += 1
            if len(messages) < max_messages:
                messages.append({"sample": i, "type": "error", "msg": "annotations must be a list"})
            continue

        for j, ann in enumerate(anns):
            total_annotations += 1
            if not isinstance(ann, dict):
                error_count += 1
                if len(messages) < max_messages:
                    messages.append({
                        "sample": i,
                        "ann_index": j,
                        "type": "error",
                        "msg": "annotation item is not a dict",
                    })
                continue

            errs, warns = validate_annotation(
                ann,
                width=w,
                height=h,
                breed_set=breed_set,
                emotion_set=emotion_set,
                action_set=action_set,
                allow_unknown_breed=allow_unknown_breed,
            )

            error_count += len(errs)
            warning_count += len(warns)

            for e in errs:
                if len(messages) < max_messages:
                    messages.append({
                        "sample": i,
                        "ann_index": j,
                        "type": "error",
                        "msg": e,
                    })
            for wmsg in warns:
                if len(messages) < max_messages:
                    messages.append({
                        "sample": i,
                        "ann_index": j,
                        "type": "warning",
                        "msg": wmsg,
                    })

    return {
        "manifest_path": str(Path(manifest_path).resolve()),
        "total_samples": total_samples,
        "total_annotations": total_annotations,
        "error_count": error_count,
        "warning_count": warning_count,
        "messages": messages,
        "is_valid": error_count == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate DogYOLO JSON labels and boxes")
    parser.add_argument("--manifest", required=True, help="Path to dataset manifest json")
    parser.add_argument("--breed-names", type=str, default=None, help="Optional text file (one breed per line)")
    parser.add_argument("--emotion-names", type=str, default=None, help="Optional text file (one emotion per line)")
    parser.add_argument("--action-names", type=str, default=None, help="Optional text file (one action per line)")
    parser.add_argument("--allow-unknown-breed", action="store_true", help="Allow Unknown breed label")
    parser.add_argument("--max-messages", type=int, default=100)
    args = parser.parse_args()

    breed_names = load_class_names(args.breed_names)
    emotion_names = load_class_names(args.emotion_names)
    action_names = load_class_names(args.action_names)

    report = validate_manifest(
        manifest_path=args.manifest,
        breed_names=breed_names if breed_names else None,
        emotion_names=emotion_names if emotion_names else None,
        action_names=action_names if action_names else None,
        allow_unknown_breed=args.allow_unknown_breed,
        max_messages=args.max_messages,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if report["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
