# Minimal Body/Head Multi-Object Detector (No Transformer)

This directory contains a minimal learning example for slot-based multi-object detection
without transformer/attention.

## What it does

- Predicts a fixed number of slots `Q` for each image
- Each slot outputs:
  - `objectness` logit
  - `body_box` in normalized `cx, cy, w, h`
  - `head_box` in normalized `cx, cy, w, h`
- Matches ground truth objects to predicted slots with greedy matching
- Trains with:
  - objectness BCE
  - SmoothL1 for body/head box regression
  - IoU loss term

## File

- `minimal_body_head_multibox.py`

## Run

```bash
python3 experiments/minimal_multibox_body_head/minimal_body_head_multibox.py
```

The script prints a sample loss and per-image prediction counts from a minimal forward/loss/infer flow.

## Target format

`multibox_body_head_loss` expects a list of length `B`:

```python
targets = [
    {
        "body_boxes": Tensor[Mi, 4],  # normalized cxcywh
        "head_boxes": Tensor[Mi, 4],  # normalized cxcywh
    },
    ...
]
```

`Mi` can vary per image, while model queries `Q` stay fixed.

## Loss diagnostics (recommended)

The loss function supports debug-friendly outputs:

```python
loss_dict = multibox_body_head_loss(
    outputs,
    targets,
    return_details=True,
    normalize_by="matched",  # prevents box loss from being diluted by batch averaging
    obj_pos_weight=1.0,      # increase (e.g. 2.0~5.0) if positives are too sparse
    box_smooth_l1_beta=0.1,  # stronger box gradients for normalized coordinates
    match_w_iou=2.0,         # bias matcher toward overlap quality
    match_w_obj=0.05,        # keep objectness from overpowering box matching
    validate_targets=True,   # range/format sanity stats for GT
)

total_loss = loss_dict["total"]
print(
    "obj:", float(loss_dict["obj"]),
    "body:", float(loss_dict["body"]),
    "head:", float(loss_dict["head"]),
    "iou:", float(loss_dict["iou"]),
    "matched:", loss_dict["num_matched"],
    "ratio:", loss_dict["matched_ratio"],
    "body_abs:", float(loss_dict["mean_abs_body_coord_err"]),
    "head_abs:", float(loss_dict["mean_abs_head_coord_err"]),
    "body_iou:", float(loss_dict["mean_body_iou"]),
    "head_iou:", float(loss_dict["mean_head_iou"]),
    "pos_obj:", float(loss_dict["mean_pos_obj_prob"]),
    "neg_obj:", float(loss_dict["mean_neg_obj_prob"]),
)
```

Set `debug=True` to print per-image matched counts and box absolute error means.
Set `strict_target_check=True` to raise an error immediately if GT boxes are outside [0,1] or invalid as `cxcywh`.
