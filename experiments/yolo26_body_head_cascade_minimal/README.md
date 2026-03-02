# YOLO26 Body->Head Cascade Minimal

This directory provides a minimal two-stage pipeline:

1. `(B, 3, H, W)` -> YOLO26 body detection
2. body ROIs are cropped/resized -> `(N, 3, Hc, Wc)`
3. head detection per ROI (top-1 or none)

## File

- `yolo26_body_head_cascade.py`

## Run

```bash
python3 experiments/yolo26_body_head_cascade_minimal/yolo26_body_head_cascade.py
```

For real YOLO26 checkpoints:

```bash
pip install ultralytics
```

## I/O Summary

Input:

- `images`: `(B, 3, H, W)`

Main outputs:

- `roi_crops`: `(N, 3, crop_h, crop_w)`
- `roi_image_indices`: `(N,)`
- `roi_body_boxes_xyxy`: `(N, 4)` absolute pixel `xyxy`
- `roi_crop_bounds_xyxy_half_open`: `(N, 4)` integer-like half-open `[x1, y1, x2, y2)`
- `roi_head_boxes_xyxy`: `(N, 4)` absolute pixel `xyxy`
- `roi_head_valid`: `(N,)` bool (`False` means no visible head)
- `per_image`: list of length `B` with grouped body/head tensors

## Notes

- Class filtering is strict by default (`strict_class_filter=True`).
- To avoid Ultralytics tensor-shape errors, input size and crop size should
  be divisible by `stride_multiple` (default `32`).
