#!/usr/bin/env python3
"""
YOLO26 Body -> ROI -> Head Cascade (Minimal)

Pipeline:
1) Input images: (B, 3, H, W)
2) Stage-1 body detector (YOLO26n recommended): detect dog body boxes
3) Crop each body ROI and resize to fixed crop size -> (N, 3, Hc, Wc)
4) Stage-2 head detector (smaller model): detect head per ROI
5) Keep top-1 head per ROI, or mark as "no head"

References:
- YOLO26 overview and usage:
  https://docs.ultralytics.com/models/yolo26/
- Ultralytics tensor inference examples:
  https://docs.ultralytics.com/modes/predict/
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - exercised only when ultralytics is absent.
    YOLO = None


DetectorInput = Union[str, Any]


def _empty_xyxy(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((0, 4), device=device, dtype=dtype)


def _empty_scores(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty((0,), device=device, dtype=dtype)


class YOLO26BodyHeadCascade(nn.Module):
    """
    Two-stage detector using Ultralytics YOLO models.

    Notes:
    - Inference-focused, intentionally simple.
    - Body detector and head detector can be model paths or injected detector objects.
    - Output boxes are absolute pixel-space `xyxy` in closed bounds.
    - `roi_crop_bounds_xyxy_half_open` is an exception and uses half-open bounds [x1, y1, x2, y2).
    """

    def __init__(
        self,
        body_model: DetectorInput = "yolo26n.pt",
        head_model: DetectorInput = "yolo26n.pt",
        crop_size: Tuple[int, int] = (224, 224),
        body_conf: float = 0.25,
        head_conf: float = 0.20,
        body_iou: float = 0.50,
        head_iou: float = 0.45,
        max_body_detections: int = 50,
        max_head_detections_per_roi: int = 1,
        head_batch_size: int = 256,
        body_class_id: Optional[int] = None,
        head_class_id: Optional[int] = None,
        body_class_name: Optional[str] = "dog",
        head_class_name: Optional[str] = None,
        strict_class_filter: bool = True,
        auto_normalize_input: bool = True,
        stride_multiple: int = 32,
        enforce_stride_multiple: bool = True,
    ) -> None:
        super().__init__()
        if crop_size[0] < 8 or crop_size[1] < 8:
            raise ValueError(f"crop_size must be >= (8, 8), got {crop_size}")
        if max_body_detections < 1:
            raise ValueError("max_body_detections must be >= 1")
        if max_head_detections_per_roi < 1:
            raise ValueError("max_head_detections_per_roi must be >= 1")
        if head_batch_size < 1:
            raise ValueError("head_batch_size must be >= 1")
        if stride_multiple < 1:
            raise ValueError("stride_multiple must be >= 1")
        for name, value in {
            "body_conf": body_conf,
            "head_conf": head_conf,
            "body_iou": body_iou,
            "head_iou": head_iou,
        }.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {value}")
        if enforce_stride_multiple and (
            crop_size[0] % stride_multiple != 0 or crop_size[1] % stride_multiple != 0
        ):
            raise ValueError(
                "crop_size must be divisible by stride_multiple when enforce_stride_multiple=True, "
                f"got crop_size={crop_size}, stride_multiple={stride_multiple}"
            )

        self.crop_size = crop_size
        self.body_conf = body_conf
        self.head_conf = head_conf
        self.body_iou = body_iou
        self.head_iou = head_iou
        self.max_body_detections = max_body_detections
        self.max_head_detections_per_roi = max_head_detections_per_roi
        self.head_batch_size = head_batch_size
        self.auto_normalize_input = auto_normalize_input
        self.strict_class_filter = strict_class_filter
        self.stride_multiple = stride_multiple
        self.enforce_stride_multiple = enforce_stride_multiple

        self.body_detector = self._build_detector(body_model)
        self.head_detector = self._build_detector(head_model)

        self.body_class_id = self._resolve_class_id(
            detector=self.body_detector,
            class_id=body_class_id,
            class_name=body_class_name,
            strict=strict_class_filter,
            role="body",
        )
        self.head_class_id = self._resolve_class_id(
            detector=self.head_detector,
            class_id=head_class_id,
            class_name=head_class_name,
            strict=strict_class_filter,
            role="head",
        )

    @staticmethod
    def _build_detector(model_or_detector: DetectorInput) -> Any:
        if isinstance(model_or_detector, str):
            if YOLO is None:
                raise ImportError(
                    "ultralytics is required when model paths are given. "
                    "Install it with: pip install ultralytics"
                )
            return YOLO(model_or_detector)
        if hasattr(model_or_detector, "predict") or callable(model_or_detector):
            return model_or_detector
        raise TypeError(
            "model_or_detector must be a model path string or an object with predict()/__call__()."
        )

    @staticmethod
    def _resolve_class_id(
        detector: Any,
        class_id: Optional[int],
        class_name: Optional[str],
        strict: bool,
        role: str,
    ) -> Optional[int]:
        if class_id is not None:
            return int(class_id)
        if class_name is None:
            return None

        target = class_name.strip().lower()
        names = getattr(detector, "names", None)
        if isinstance(names, dict):
            for idx, name in names.items():
                if str(name).strip().lower() == target:
                    return int(idx)
        if isinstance(names, (list, tuple)):
            for idx, name in enumerate(names):
                if str(name).strip().lower() == target:
                    return int(idx)

        if strict:
            raise ValueError(
                f"{role}_class_name='{class_name}' not found in detector names. "
                f"Provide explicit {role}_class_id or disable strict_class_filter."
            )
        return None

    def _validate_stride(self, height: int, width: int, stage_name: str) -> None:
        if not self.enforce_stride_multiple:
            return
        if (height % self.stride_multiple) != 0 or (width % self.stride_multiple) != 0:
            raise ValueError(
                f"{stage_name} size {(height, width)} must be divisible by {self.stride_multiple}. "
                "Ultralytics tensor inference expects stride-aligned sizes."
            )

    @staticmethod
    def _prepare_images(images: torch.Tensor, auto_normalize_input: bool) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"images must be (B, 3, H, W), got {tuple(images.shape)}")
        if images.shape[0] < 1:
            raise ValueError("batch size must be >= 1")
        if images.shape[2] < 1 or images.shape[3] < 1:
            raise ValueError(f"image height/width must be >= 1, got {tuple(images.shape[2:4])}")

        x = images
        if not x.is_floating_point():
            x = x.float() / 255.0
        else:
            if x.dtype == torch.float64:
                x = x.float()
            if auto_normalize_input:
                # Ultralytics tensor inputs are expected in [0, 1].
                x_max = float(x.detach().float().max().item())
                if x_max > 1.5:
                    x = x / 255.0

        if x.device.type == "cpu" and x.dtype in (torch.float16, torch.bfloat16):
            x = x.float()

        if not torch.isfinite(x).all():
            raise ValueError("images contain NaN/Inf")
        return x.contiguous()

    @staticmethod
    def _run_detector(
        detector: Any,
        images: torch.Tensor,
        conf: float,
        iou: float,
        max_det: int,
    ) -> Sequence[Any]:
        kwargs = {
            "source": images,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "verbose": False,
        }
        if hasattr(detector, "predict"):
            return detector.predict(**kwargs)
        if callable(detector):
            try:
                return detector(**kwargs)
            except TypeError:
                return detector(images)
        raise TypeError("detector must provide predict() or be callable.")

    @staticmethod
    def _parse_result_boxes(
        result: Any,
        *,
        device: torch.device,
        dtype: torch.dtype,
        class_id: Optional[int],
        score_thresh: float,
        max_det: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        boxes = getattr(result, "boxes", None)
        if boxes is None or getattr(boxes, "xyxy", None) is None:
            return _empty_xyxy(device, dtype), _empty_scores(device, dtype)

        xyxy = boxes.xyxy.to(device=device, dtype=dtype)
        conf = boxes.conf.to(device=device, dtype=dtype)
        cls = boxes.cls.to(device=device, dtype=torch.long)
        if xyxy.numel() == 0:
            return _empty_xyxy(device, dtype), _empty_scores(device, dtype)

        keep = conf >= float(score_thresh)
        if class_id is not None:
            keep &= cls == int(class_id)

        xyxy = xyxy[keep]
        conf = conf[keep]
        if xyxy.numel() == 0:
            return _empty_xyxy(device, dtype), _empty_scores(device, dtype)

        k = min(max_det, conf.shape[0])
        if k < conf.shape[0]:
            conf_topk, idx_topk = torch.topk(conf, k=k, largest=True, sorted=True)
            return xyxy[idx_topk], conf_topk

        order = torch.argsort(conf, descending=True)
        return xyxy[order], conf[order]

    @staticmethod
    def _boxes_intersect_image(boxes: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return torch.zeros((0,), dtype=torch.bool, device=boxes.device)
        return (
            (boxes[:, 2] > 0.0)
            & (boxes[:, 3] > 0.0)
            & (boxes[:, 0] < float(image_w))
            & (boxes[:, 1] < float(image_h))
        )

    @staticmethod
    def _clip_xyxy_closed(boxes: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
        if boxes.numel() == 0:
            return boxes
        max_x = float(max(0, image_w - 1))
        max_y = float(max(0, image_h - 1))
        x1 = boxes[:, 0].clamp(0.0, max_x)
        y1 = boxes[:, 1].clamp(0.0, max_y)
        x2 = boxes[:, 2].clamp(0.0, max_x)
        y2 = boxes[:, 3].clamp(0.0, max_y)
        left = torch.minimum(x1, x2)
        top = torch.minimum(y1, y2)
        right = torch.maximum(x1, x2)
        bottom = torch.maximum(y1, y2)
        return torch.stack([left, top, right, bottom], dim=1)

    @staticmethod
    def _to_crop_bounds_from_closed(boxes: torch.Tensor, image_h: int, image_w: int) -> torch.Tensor:
        """
        Convert closed xyxy boxes to half-open integer slicing bounds [x1, y1, x2, y2).
        """
        if boxes.numel() == 0:
            return torch.empty((0, 4), dtype=torch.long, device=boxes.device)

        x1i = torch.floor(boxes[:, 0]).to(torch.long).clamp(0, max(0, image_w - 1))
        y1i = torch.floor(boxes[:, 1]).to(torch.long).clamp(0, max(0, image_h - 1))
        x2i = (torch.floor(boxes[:, 2]).to(torch.long) + 1).clamp(1, max(1, image_w))
        y2i = (torch.floor(boxes[:, 3]).to(torch.long) + 1).clamp(1, max(1, image_h))

        x2i = torch.maximum(x2i, x1i + 1)
        y2i = torch.maximum(y2i, y1i + 1)
        return torch.stack([x1i, y1i, x2i, y2i], dim=1)

    @staticmethod
    def _crop_bounds_to_closed_xyxy(crop_bounds: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if crop_bounds.numel() == 0:
            return torch.empty((0, 4), dtype=dtype, device=crop_bounds.device)
        x1 = crop_bounds[:, 0].to(dtype=dtype)
        y1 = crop_bounds[:, 1].to(dtype=dtype)
        x2 = (crop_bounds[:, 2] - 1).to(dtype=dtype)
        y2 = (crop_bounds[:, 3] - 1).to(dtype=dtype)
        return torch.stack([x1, y1, x2, y2], dim=1)

    def _crop_single_roi(self, image_chw: torch.Tensor, crop_bounds: torch.Tensor) -> torch.Tensor:
        x1i, y1i, x2i, y2i = [int(v) for v in crop_bounds.tolist()]
        patch = image_chw[:, y1i:y2i, x1i:x2i].unsqueeze(0)
        return F.interpolate(patch, size=self.crop_size, mode="bilinear", align_corners=False).squeeze(0)

    def _map_head_crop_box_to_image(
        self,
        head_box_crop_xyxy: torch.Tensor,
        crop_bounds_xyxy: torch.Tensor,
        image_h: int,
        image_w: int,
    ) -> torch.Tensor:
        crop_h, crop_w = self.crop_size
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_bounds_xyxy.to(dtype=head_box_crop_xyxy.dtype)
        body_w = (crop_x2 - crop_x1).clamp(min=1.0)
        body_h = (crop_y2 - crop_y1).clamp(min=1.0)
        sx = body_w / float(crop_w)
        sy = body_h / float(crop_h)

        hx1, hy1, hx2, hy2 = head_box_crop_xyxy
        gx1 = crop_x1 + hx1 * sx
        gy1 = crop_y1 + hy1 * sy
        gx2 = crop_x1 + hx2 * sx
        gy2 = crop_y1 + hy2 * sy
        out = torch.stack([gx1, gy1, gx2, gy2], dim=0).unsqueeze(0)
        return self._clip_xyxy_closed(out, image_h=image_h, image_w=image_w).squeeze(0)

    def _empty_output(
        self,
        batch_size: int,
        device: torch.device,
        image_dtype: torch.dtype,
        box_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        empty_roi = torch.empty(
            (0, 3, self.crop_size[0], self.crop_size[1]),
            device=device,
            dtype=image_dtype,
        )
        empty_indices = torch.empty((0,), dtype=torch.long, device=device)
        empty_boxes = _empty_xyxy(device, box_dtype)
        empty_scores = _empty_scores(device, box_dtype)
        empty_valid = torch.empty((0,), dtype=torch.bool, device=device)
        empty_bounds = torch.empty((0, 4), dtype=torch.long, device=device)

        per_image = [
            {
                "body_boxes_xyxy": _empty_xyxy(device, box_dtype),
                "body_scores": _empty_scores(device, box_dtype),
                "head_boxes_xyxy": _empty_xyxy(device, box_dtype),
                "head_scores": _empty_scores(device, box_dtype),
                "head_valid": torch.empty((0,), dtype=torch.bool, device=device),
            }
            for _ in range(batch_size)
        ]
        return {
            "roi_crops": empty_roi,
            "roi_image_indices": empty_indices,
            "roi_body_boxes_xyxy": empty_boxes,
            "roi_body_scores": empty_scores,
            "roi_crop_bounds_xyxy_half_open": empty_bounds,
            # backward-compatible alias
            "roi_crop_bounds_xyxy": empty_bounds,
            "roi_head_boxes_xyxy": empty_boxes,
            "roi_head_scores": empty_scores,
            "roi_head_valid": empty_valid,
            # aliases for convenience
            "head_boxes_xyxy": empty_boxes,
            "head_scores": empty_scores,
            "head_valid": empty_valid,
            "per_image": per_image,
        }

    def forward(self, images: torch.Tensor) -> Dict[str, Any]:
        x = self._prepare_images(images, auto_normalize_input=self.auto_normalize_input)
        batch_size, _, image_h, image_w = x.shape
        self._validate_stride(image_h, image_w, stage_name="input")

        device = x.device
        image_dtype = x.dtype
        box_dtype = torch.float32

        body_results = self._run_detector(
            detector=self.body_detector,
            images=x,
            conf=self.body_conf,
            iou=self.body_iou,
            max_det=self.max_body_detections,
        )
        if len(body_results) != batch_size:
            raise ValueError(
                f"body detector returned {len(body_results)} results for batch size {batch_size}"
            )

        roi_crops_list: List[torch.Tensor] = []
        roi_image_indices_list: List[int] = []
        roi_body_boxes_list: List[torch.Tensor] = []
        roi_body_scores_list: List[torch.Tensor] = []
        roi_crop_bounds_list: List[torch.Tensor] = []
        roi_counts_per_image = [0 for _ in range(batch_size)]

        for image_idx in range(batch_size):
            body_boxes, body_scores = self._parse_result_boxes(
                body_results[image_idx],
                device=device,
                dtype=box_dtype,
                class_id=self.body_class_id,
                score_thresh=self.body_conf,
                max_det=self.max_body_detections,
            )
            if body_boxes.numel() == 0:
                continue

            keep_inside = self._boxes_intersect_image(body_boxes, image_h=image_h, image_w=image_w)
            body_boxes = body_boxes[keep_inside]
            body_scores = body_scores[keep_inside]
            if body_boxes.numel() == 0:
                continue

            body_boxes = self._clip_xyxy_closed(body_boxes, image_h=image_h, image_w=image_w)
            crop_bounds = self._to_crop_bounds_from_closed(body_boxes, image_h=image_h, image_w=image_w)
            body_boxes_aligned = self._crop_bounds_to_closed_xyxy(crop_bounds, dtype=box_dtype)

            for body_idx in range(crop_bounds.shape[0]):
                roi_crops_list.append(self._crop_single_roi(x[image_idx], crop_bounds[body_idx]))
                roi_image_indices_list.append(image_idx)
                roi_body_boxes_list.append(body_boxes_aligned[body_idx])
                roi_body_scores_list.append(body_scores[body_idx])
                roi_crop_bounds_list.append(crop_bounds[body_idx])
                roi_counts_per_image[image_idx] += 1

        if not roi_crops_list:
            return self._empty_output(
                batch_size=batch_size,
                device=device,
                image_dtype=image_dtype,
                box_dtype=box_dtype,
            )

        roi_crops = torch.stack(roi_crops_list, dim=0).to(dtype=image_dtype)
        roi_count = roi_crops.shape[0]

        roi_image_indices = torch.tensor(roi_image_indices_list, dtype=torch.long, device=device)
        roi_body_boxes = torch.stack(roi_body_boxes_list, dim=0).to(device=device, dtype=box_dtype)
        roi_body_scores = torch.stack(roi_body_scores_list, dim=0).to(device=device, dtype=box_dtype)
        roi_crop_bounds = torch.stack(roi_crop_bounds_list, dim=0).to(device=device, dtype=torch.long)

        head_results_all: List[Any] = []
        for start in range(0, roi_count, self.head_batch_size):
            end = min(start + self.head_batch_size, roi_count)
            head_results_chunk = self._run_detector(
                detector=self.head_detector,
                images=roi_crops[start:end],
                conf=self.head_conf,
                iou=self.head_iou,
                max_det=self.max_head_detections_per_roi,
            )
            if len(head_results_chunk) != (end - start):
                raise ValueError(
                    f"head detector returned {len(head_results_chunk)} results for chunk size {end - start}"
                )
            head_results_all.extend(head_results_chunk)
        if len(head_results_all) != roi_count:
            raise ValueError(
                f"head detector returned {len(head_results_all)} total results for ROI count {roi_count}"
            )

        roi_head_boxes = torch.zeros((roi_count, 4), dtype=box_dtype, device=device)
        roi_head_scores = torch.zeros((roi_count,), dtype=box_dtype, device=device)
        roi_head_valid = torch.zeros((roi_count,), dtype=torch.bool, device=device)

        for roi_idx in range(roi_count):
            head_boxes_crop, head_scores_crop = self._parse_result_boxes(
                head_results_all[roi_idx],
                device=device,
                dtype=box_dtype,
                class_id=self.head_class_id,
                score_thresh=self.head_conf,
                max_det=self.max_head_detections_per_roi,
            )
            if head_boxes_crop.numel() == 0:
                continue

            best_head_crop = head_boxes_crop[0]
            mapped_head = self._map_head_crop_box_to_image(
                head_box_crop_xyxy=best_head_crop,
                crop_bounds_xyxy=roi_crop_bounds[roi_idx],
                image_h=image_h,
                image_w=image_w,
            )
            roi_head_boxes[roi_idx] = mapped_head
            roi_head_scores[roi_idx] = head_scores_crop[0]
            roi_head_valid[roi_idx] = True

        per_image: List[Dict[str, torch.Tensor]] = []
        start = 0
        for image_idx in range(batch_size):
            count = roi_counts_per_image[image_idx]
            end = start + count
            per_image.append(
                {
                    "body_boxes_xyxy": roi_body_boxes[start:end],
                    "body_scores": roi_body_scores[start:end],
                    "head_boxes_xyxy": roi_head_boxes[start:end],
                    "head_scores": roi_head_scores[start:end],
                    "head_valid": roi_head_valid[start:end],
                }
            )
            start = end

        return {
            "roi_crops": roi_crops,
            "roi_image_indices": roi_image_indices,
            "roi_body_boxes_xyxy": roi_body_boxes,
            "roi_body_scores": roi_body_scores,
            "roi_crop_bounds_xyxy_half_open": roi_crop_bounds,
            # backward-compatible alias
            "roi_crop_bounds_xyxy": roi_crop_bounds,
            "roi_head_boxes_xyxy": roi_head_boxes,
            "roi_head_scores": roi_head_scores,
            "roi_head_valid": roi_head_valid,
            # aliases for convenience
            "head_boxes_xyxy": roi_head_boxes,
            "head_scores": roi_head_scores,
            "head_valid": roi_head_valid,
            "per_image": per_image,
        }


class _FakeBoxes:
    def __init__(self, xyxy: torch.Tensor, conf: torch.Tensor, cls: torch.Tensor) -> None:
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls


class _FakeResult:
    def __init__(self, boxes: _FakeBoxes) -> None:
        self.boxes = boxes


class _FakeBodyDetector:
    names = {0: "dog"}

    def predict(self, source: torch.Tensor, **_: Any) -> List[_FakeResult]:
        bsz, _, h, w = source.shape
        results: List[_FakeResult] = []
        for idx in range(bsz):
            if idx % 2 == 0:
                xyxy = torch.tensor([[0.15 * w, 0.10 * h, 0.80 * w, 0.95 * h]], dtype=source.dtype)
                conf = torch.tensor([0.92], dtype=source.dtype)
                cls = torch.tensor([0], dtype=torch.long)
            else:
                xyxy = torch.tensor(
                    [
                        [0.10 * w, 0.08 * h, 0.55 * w, 0.85 * h],
                        [0.52 * w, 0.12 * h, 0.92 * w, 0.88 * h],
                    ],
                    dtype=source.dtype,
                )
                conf = torch.tensor([0.90, 0.87], dtype=source.dtype)
                cls = torch.tensor([0, 0], dtype=torch.long)
            results.append(_FakeResult(_FakeBoxes(xyxy=xyxy, conf=conf, cls=cls)))
        return results


class _FakeHeadDetector:
    names = {0: "head"}

    def predict(self, source: torch.Tensor, **_: Any) -> List[_FakeResult]:
        bsz, _, h, w = source.shape
        results: List[_FakeResult] = []
        for idx in range(bsz):
            if idx % 2 == 0:
                xyxy = torch.tensor([[0.35 * w, 0.08 * h, 0.70 * w, 0.35 * h]], dtype=source.dtype)
                conf = torch.tensor([0.89], dtype=source.dtype)
                cls = torch.tensor([0], dtype=torch.long)
            else:
                xyxy = torch.empty((0, 4), dtype=source.dtype)
                conf = torch.empty((0,), dtype=source.dtype)
                cls = torch.empty((0,), dtype=torch.long)
            results.append(_FakeResult(_FakeBoxes(xyxy=xyxy, conf=conf, cls=cls)))
        return results


if __name__ == "__main__":
    model = YOLO26BodyHeadCascade(
        body_model=_FakeBodyDetector(),
        head_model=_FakeHeadDetector(),
        crop_size=(160, 160),
        body_class_name="dog",
        head_class_name="head",
        strict_class_filter=True,
    )
    images = torch.rand(2, 3, 256, 256)
    outputs = model(images)

    assert outputs["roi_crops"].shape[0] == outputs["roi_body_boxes_xyxy"].shape[0]
    assert outputs["roi_head_boxes_xyxy"].shape == outputs["roi_body_boxes_xyxy"].shape
    assert outputs["roi_head_valid"].dtype == torch.bool

    print("roi_crops:", tuple(outputs["roi_crops"].shape))
    print("roi_body_boxes_xyxy:", tuple(outputs["roi_body_boxes_xyxy"].shape))
    print("roi_head_boxes_xyxy:", tuple(outputs["roi_head_boxes_xyxy"].shape))
    print("roi_head_valid:", outputs["roi_head_valid"].tolist())
    for idx, item in enumerate(outputs["per_image"]):
        print(
            f"image[{idx}] bodies={item['body_boxes_xyxy'].shape[0]} "
            f"heads_visible={int(item['head_valid'].sum().item())}"
        )
