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
python experiments/minimal_multibox_body_head/minimal_body_head_multibox.py
```

Expected output shape details are printed by the script through a minimal forward/loss/infer flow.

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
