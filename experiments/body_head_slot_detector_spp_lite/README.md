# Body/Head Slot Detector SPP-Lite

Experimental cascaded body/head slot detector with SPP-Lite body pooling and a compressed head ROI stage.

## Files

- `cascaded_body_head_model_spp_lite.py` (single source of truth)

## Compatibility Aliases

To minimize import-path changes, this module also exposes:

- `CascadedBodyHeadDetector` -> `CascadedBodyHeadDetectorSPPLite`
- `cascaded_body_head_loss` -> `cascaded_body_head_loss_spp_lite`

## Model Upgrades

- Body branch uses SPP-Lite (`1x1 + 2x2 + 4x4`) instead of GAP only.
- Head branch uses ROIAlign `5x5` and a smaller MLP.
- Optional Tiny48 CFF-like context fusion for ROI head features (`use_tiny48_cff=True`).

## Loss Utilities in File

- `greedy_match_slots_split`: split body/head/objectness matching cost.
  - Optional extras: geometric term (`geo_w`) and head precision hinge.
- `body_head_set_loss_spp_lite`: set loss with optional geometric consistency loss (`w_geo`).
- `body_head_set_loss_with_curriculum`: schedules `w_body` and `w_head` over `step/total_steps`.

Recommended stronger knobs are provided as constants:

- `RECOMMENDED_MATCH_EXTRAS`
- `RECOMMENDED_CURRICULUM`

Defaults keep optional extras conservative/off.

Current curriculum default follows:
- early: `w_body=8, w_head=10`
- late: `w_body=4, w_head=20`

## Output Contract

`forward` returns:

- `pred_logits`: `(B, Q)`
- `pred_body_boxes`: `(B, Q, 4)` normalized `cx, cy, w, h`
- `pred_head_boxes`: `(B, Q, 4)` normalized `cx, cy, w, h`

## Quick Run

```bash
python3 experiments/body_head_slot_detector_spp_lite/cascaded_body_head_model_spp_lite.py
```
