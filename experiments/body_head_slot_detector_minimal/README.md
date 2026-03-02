# Body/Head Slot Detector Minimal (No Transformer)

This directory is an isolated study implementation for multi-object detection
with fixed slots and matching.

## Goal

- Keep the model minimal and easy to read from `forward`.
- Predict **body box + head box** per slot.
- Train with slot matching and no-object supervision.
- Use a simple split path:
  - body from global pooled feature
  - head from per-body ROI feature (tiny head-only backbone)

## File

- `body_head_slot_model.py`
- `cascaded_body_head_model.py`
- YOLO26 cascade variant: `experiments/yolo26_body_head_cascade_minimal/`

## Run

Prerequisites:

- Python 3.10+ (tested in local `.venv`)
- PyTorch installed in the environment

```bash
python3 experiments/body_head_slot_detector_minimal/body_head_slot_model.py
```

## I/O Summary

Input:

- image tensor `(B, 3, H, W)`

Output from model:

- `pred_logits`: `(B, Q)` objectness logits
- `pred_body_boxes`: `(B, Q, 4)` normalized `cx,cy,w,h`
- `pred_head_boxes`: `(B, Q, 4)` normalized `cx,cy,w,h`

Model args:

- `num_queries` (default: `50`)
- `roi_size` (default: `5`) for ROI head feature size

Target format (list length `B`):

```python
targets = [
    {
        "body_boxes": Tensor[Mi, 4],  # normalized cxcywh
        "head_boxes": Tensor[Mi, 4],  # normalized cxcywh
    },
    ...
]
```

Constraint:

- Per image object count `Mi` must satisfy `Mi <= Q` (`Q=num_queries`).
- If `Mi > Q`, loss raises an error to avoid silent GT dropping.

## Debug-Friendly Loss

`body_head_set_loss(..., return_details=True)` returns:

- `obj`, `body`, `head`, `iou`
- `num_matched`, `matched_ratio`, `matched_per_image`
- `mean_abs_body_coord_err`, `mean_abs_head_coord_err`
- `mean_body_iou`, `mean_head_iou`
- `mean_pos_obj_prob`, `mean_neg_obj_prob`

Use `validate_targets=True` for target sanity stats.
Use `strict_target_check=True` to fail fast on invalid `cxcywh` targets.
Matching is intentionally greedy for study simplicity (not Hungarian-optimal).
