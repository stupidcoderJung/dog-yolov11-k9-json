#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import ImageReadMode, read_image


Annotation = Dict[str, Any]
Sample = Dict[str, Any]


def xyxy_to_coco(box: Sequence[float]) -> List[int]:
    x1, y1, x2, y2 = box
    return [
        int(round(x1)),
        int(round(y1)),
        int(round(max(0.0, x2 - x1))),
        int(round(max(0.0, y2 - y1))),
    ]


def load_class_names(path: Optional[str]) -> List[str]:
    if path is None:
        return []
    p = Path(path)
    names: List[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        name = line.strip()
        if not name or name.startswith("#"):
            continue
        names.append(name)
    return names


def load_manifest(manifest_path: str) -> List[Sample]:
    p = Path(manifest_path)
    raw = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(raw, dict):
        if "samples" in raw and isinstance(raw["samples"], list):
            raw = raw["samples"]
        else:
            raise ValueError("Manifest dict must contain a list under 'samples'")

    if not isinstance(raw, list):
        raise ValueError("Manifest must be a list or a dict with 'samples'")

    out: List[Sample] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"Manifest item #{idx} must be a dict")
        anns = item.get("annotations", [])
        if not isinstance(anns, list):
            raise ValueError(f"Manifest item #{idx} has non-list annotations")
        out.append(item)
    return out


def _resolve_image_path(sample: Sample, manifest_dir: Path) -> Path:
    image_key_candidates = ["image", "image_path", "image_file", "file_name"]
    image_value = None
    for k in image_key_candidates:
        if k not in sample:
            continue
        raw = sample.get(k, None)
        if raw is None:
            continue
        v = str(raw).strip()
        if not v:
            continue
        if v.lower() in {"none", "null"}:
            continue
        image_value = v
        break
    if image_value is None:
        raise ValueError("Sample does not have image path key (image/image_path/image_file/file_name)")

    p = Path(str(image_value))
    if p.is_absolute():
        return p
    return (manifest_dir / p).resolve()


def _scale_xyxy(box: Sequence[float], sx: float, sy: float) -> List[int]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    ]


def resize_annotations(
    annotations: List[Annotation],
    scale_x: float,
    scale_y: float,
) -> List[Annotation]:
    resized: List[Annotation] = []
    for ann in annotations:
        out = copy.deepcopy(ann)

        body = out.get("bodybndbox", None)
        if isinstance(body, list) and len(body) == 4:
            body_xyxy = _scale_xyxy(body, scale_x, scale_y)
            out["bodybndbox"] = body_xyxy
            out["bodybndbox_coco"] = xyxy_to_coco(body_xyxy)

        head = out.get("headbndbox", None)
        if isinstance(head, list) and len(head) == 4:
            head_xyxy = _scale_xyxy(head, scale_x, scale_y)
            out["headbndbox"] = head_xyxy
            out["headbndbox_coco"] = xyxy_to_coco(head_xyxy)

        resized.append(out)
    return resized


def collect_unique_values(samples: List[Sample], key: str) -> List[str]:
    seen = set()
    values: List[str] = []
    for sample in samples:
        anns = sample.get("annotations", [])
        if not isinstance(anns, list):
            continue
        for ann in anns:
            if not isinstance(ann, dict):
                continue
            raw = ann.get(key, None)
            if raw is None:
                continue
            v = str(raw).strip()
            if not v:
                continue
            if v in seen:
                continue
            seen.add(v)
            values.append(v)
    return values


class DogJsonDataset(Dataset):
    """
    Manifest format (list or {'samples': list}):
      {
        "image": "images/a.jpg",
        "annotations": [
          {
            "label": "Border Collie",
            "bodybndbox": [x1, y1, x2, y2],
            "headbndbox": [x1, y1, x2, y2],
            "emotional": "excited",
            "action": "running"
          }
        ]
      }
    """

    def __init__(
        self,
        manifest_path: str,
        image_size: Tuple[int, int] = (640, 640),
        normalize: bool = True,
        strict_image_exists: bool = True,
    ):
        super().__init__()
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest_dir = self.manifest_path.parent
        self.samples = load_manifest(str(self.manifest_path))
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.normalize = normalize
        self.strict_image_exists = strict_image_exists

        if len(self.samples) == 0:
            raise ValueError("Dataset manifest is empty")

        if self.strict_image_exists:
            for i, s in enumerate(self.samples):
                p = _resolve_image_path(s, self.manifest_dir)
                if not p.exists():
                    raise FileNotFoundError(f"Missing image for sample #{i}: {p}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        img_path = _resolve_image_path(sample, self.manifest_dir)

        img = read_image(str(img_path), mode=ImageReadMode.RGB).float()  # [C,H,W]
        _, h0, w0 = img.shape
        h1, w1 = self.image_size

        anns = sample.get("annotations", [])
        if not isinstance(anns, list):
            anns = []

        if h0 != h1 or w0 != w1:
            img = F.interpolate(
                img.unsqueeze(0),
                size=(h1, w1),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            sx = float(w1) / float(w0)
            sy = float(h1) / float(h0)
            anns = resize_annotations(anns, sx, sy)
        else:
            anns = copy.deepcopy(anns)

        if self.normalize:
            img = img / 255.0

        return {
            "image": img,
            "annotations": anns,
            "image_path": str(img_path),
            "orig_size": (int(h0), int(w0)),
            "image_size": (int(h1), int(w1)),
        }


def dog_collate_fn(
    batch: List[Dict[str, Any]],
) -> Tuple[torch.Tensor, List[List[Annotation]], Dict[str, Any]]:
    images = torch.stack([x["image"] for x in batch], dim=0)
    ann_lists = [x["annotations"] for x in batch]
    meta = {
        "image_paths": [x["image_path"] for x in batch],
        "orig_sizes": [x["orig_size"] for x in batch],
        "image_sizes": [x["image_size"] for x in batch],
    }
    return images, ann_lists, meta
