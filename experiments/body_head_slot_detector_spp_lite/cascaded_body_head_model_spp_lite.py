#!/usr/bin/env python3
"""
Cascaded Body/Head slot detector with SPP-Lite body pooling and compact ROI head.

Upgrades vs. the minimal cascaded model:
1) Body GAP -> SPP-Lite pooling (1x1 + 2x2 + 4x4)
2) Head ROI compression (ROIAlign 5x5 + smaller MLP)
3) Loss curriculum wrapper (early body-heavy -> late head-heavy)
4) Optional Tiny48 CFF-like context fusion on ROI features
5) Optional geometric consistency auxiliary loss
6) Greedy matcher with split body/head/objectness costs (+ optional extras)

Defaults keep optional extras conservative/off for stable behavior.

Recommended presets to try:
- RECOMMENDED_MATCH_EXTRAS
- RECOMMENDED_CURRICULUM
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover
    roi_align = None


RECOMMENDED_MATCH_EXTRAS = {
    "match_geo_w": 0.20,
    "match_head_precision_hinge_w": 1.25,
    "match_head_precision_thresh": 0.60,
    "w_geo": 0.50,
}

RECOMMENDED_CURRICULUM = {
    "w_body_start": 8.0,
    "w_body_end": 4.0,
    "w_head_start": 10.0,
    "w_head_end": 20.0,
    "curriculum_power": 1.0,
}

__all__ = [
    "CascadedBodyHeadDetectorSPPLite",
    "CascadedBodyHeadDetector",
    "body_head_set_loss_spp_lite",
    "body_head_set_loss_with_curriculum",
    "cascaded_body_head_loss_spp_lite",
    "cascaded_body_head_loss",
    "infer_slots",
    "RECOMMENDED_MATCH_EXTRAS",
    "RECOMMENDED_CURRICULUM",
]


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = boxes.unbind(-1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))

    a = a[:, None, :]
    b = b[None, :, :]

    inter_x1 = torch.maximum(a[..., 0], b[..., 0])
    inter_y1 = torch.maximum(a[..., 1], b[..., 1])
    inter_x2 = torch.minimum(a[..., 2], b[..., 2])
    inter_y2 = torch.minimum(a[..., 3], b[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = (area_a + area_b - inter).clamp(min=1e-9)
    return inter / union


def head_body_geometry(body_boxes: torch.Tensor, head_boxes: torch.Tensor) -> torch.Tensor:
    """
    Relative head geometry encoded in body coordinates.
    Returns (N, 4): [rel_cx, rel_cy, rel_w, rel_h].
    """
    if body_boxes.numel() == 0:
        return body_boxes.new_zeros((0, 4))

    body_xyxy = cxcywh_to_xyxy(body_boxes)
    bx1, by1, bx2, by2 = body_xyxy.unbind(-1)
    bw = (bx2 - bx1).clamp(min=1e-6)
    bh = (by2 - by1).clamp(min=1e-6)

    hc, hy, hw, hh = head_boxes.unbind(-1)
    rel = torch.stack(
        [
            (hc - bx1) / bw,
            (hy - by1) / bh,
            hw / bw,
            hh / bh,
        ],
        dim=-1,
    )
    return rel.clamp(min=-2.0, max=2.0)


def _as_box_tensor(boxes, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(boxes, torch.Tensor):
        out = boxes.to(device=device, dtype=dtype)
    else:
        out = torch.tensor(boxes, device=device, dtype=dtype)
    if out.numel() == 0:
        return out.reshape(0, 4)
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError(f"Expected Nx4 boxes, got shape={tuple(out.shape)}")
    return out


class SPPLite(nn.Module):
    def __init__(self, levels: Sequence[int] = (1, 2, 4)):
        super().__init__()
        if len(levels) == 0:
            raise ValueError("levels must be non-empty")
        parsed = tuple(int(level) for level in levels)
        if any(level < 1 for level in parsed):
            raise ValueError("all levels must be >= 1")
        self.levels = parsed
        self.multiplier = int(sum(level * level for level in self.levels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [F.adaptive_avg_pool2d(x, output_size=(level, level)).flatten(1) for level in self.levels]
        return torch.cat(parts, dim=1)


class Tiny48CFFLite(nn.Module):
    """
    Compact CFF-like ROI fusion:
    mix local ROI feature with image-level context of the same image.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.local_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.context_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.mix = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, roi_feat: torch.Tensor, roi_context: torch.Tensor) -> torch.Tensor:
        h, w = roi_feat.shape[-2], roi_feat.shape[-1]
        local = self.local_proj(roi_feat)
        context = self.context_proj(roi_context).expand(-1, -1, h, w)
        fused = self.mix(torch.cat([local, context], dim=1))
        return roi_feat + fused * self.gate(fused)


class CascadedBodyHeadDetectorSPPLite(nn.Module):
    """
    Cascaded 2-stage detector:
    - Stage 1: body/objectness from SPP-Lite pooled feature
    - Stage 2: head from body ROIs with compact head MLP

    Output contract:
      pred_logits: (B, Q)
      pred_body_boxes: (B, Q, 4)
      pred_head_boxes: (B, Q, 4)
    """

    def __init__(
        self,
        num_queries: int = 10,
        backbone_type: str = "tiny48",
        use_pretrained: bool = False,
        roi_size: int = 5,
        spp_levels: Sequence[int] = (1, 2, 4),
        use_tiny48_cff: bool = False,
    ):
        super().__init__()
        if num_queries < 1:
            raise ValueError("num_queries must be >= 1")
        if roi_size < 2:
            raise ValueError("roi_size must be >= 2")

        self.num_queries = int(num_queries)
        self.roi_size = int(roi_size)

        if backbone_type == "resnet18":
            weights = ResNet18_Weights.DEFAULT if use_pretrained else None
            resnet = resnet18(weights=weights)
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])
            self.feature_dim = 512
        elif backbone_type == "tiny48":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 32, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 48, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Conv2d(48, 48, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(48),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=0.10),
            )
            self.feature_dim = 48
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        self.spp = SPPLite(levels=spp_levels)
        body_in_dim = self.feature_dim * self.spp.multiplier

        body_hidden = 256 if self.feature_dim >= 128 else 192
        self.body_objectness = nn.Sequential(
            nn.Linear(body_in_dim, body_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(body_hidden, self.num_queries),
        )
        self.body_regressor = nn.Sequential(
            nn.Linear(body_in_dim, body_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(body_hidden, self.num_queries * 4),
        )

        head_in_dim = self.feature_dim * self.roi_size * self.roi_size
        head_hidden_1 = 224 if self.feature_dim >= 128 else 160
        head_hidden_2 = 96 if self.feature_dim >= 128 else 64
        self.head_regressor = nn.Sequential(
            nn.Linear(head_in_dim, head_hidden_1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(head_hidden_1, head_hidden_2),
            nn.ReLU(inplace=True),
            nn.Linear(head_hidden_2, 4),
        )

        self.use_tiny48_cff = bool(use_tiny48_cff and backbone_type == "tiny48")
        self.head_context_fusion = Tiny48CFFLite(self.feature_dim) if self.use_tiny48_cff else None

    @staticmethod
    def _decode_head_relative(body_boxes: torch.Tensor, head_relative: torch.Tensor) -> torch.Tensor:
        body_xyxy = cxcywh_to_xyxy(body_boxes)
        body_x1 = body_xyxy[..., 0]
        body_y1 = body_xyxy[..., 1]
        body_w = body_boxes[..., 2].clamp(min=1e-4)
        body_h = body_boxes[..., 3].clamp(min=1e-4)

        rel_cx, rel_cy, rel_w, rel_h = head_relative.unbind(-1)
        head_cx = body_x1 + rel_cx * body_w
        head_cy = body_y1 + rel_cy * body_h
        head_w = rel_w * body_w
        head_h = rel_h * body_h
        return torch.stack([head_cx, head_cy, head_w, head_h], dim=-1).clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        if roi_align is None:
            raise RuntimeError(
                "torchvision.ops.roi_align is not available. Install torchvision with ops support."
            )

        batch_size = x.shape[0]
        feat = self.backbone(x)

        pooled = self.spp(feat)
        pred_logits = self.body_objectness(pooled)
        pred_body_boxes = self.body_regressor(pooled).view(batch_size, self.num_queries, 4).sigmoid()

        feat_h, feat_w = feat.shape[2], feat.shape[3]
        body_xyxy = cxcywh_to_xyxy(pred_body_boxes)
        body_xyxy_feat = body_xyxy.clone()
        body_xyxy_feat[..., [0, 2]] *= float(feat_w)
        body_xyxy_feat[..., [1, 3]] *= float(feat_h)

        eps = 1e-3
        x1 = body_xyxy_feat[..., 0].clamp(0.0, max(float(feat_w) - eps, 0.0))
        y1 = body_xyxy_feat[..., 1].clamp(0.0, max(float(feat_h) - eps, 0.0))
        x2 = body_xyxy_feat[..., 2].clamp(0.0, float(feat_w))
        y2 = body_xyxy_feat[..., 3].clamp(0.0, float(feat_h))
        x2 = torch.maximum(x2, x1 + eps).clamp(max=float(feat_w))
        y2 = torch.maximum(y2, y1 + eps).clamp(max=float(feat_h))
        body_xyxy_feat = torch.stack([x1, y1, x2, y2], dim=-1)

        roi_batch_idx = (
            torch.arange(batch_size, dtype=torch.float32, device=x.device)
            .view(batch_size, 1, 1)
            .expand(batch_size, self.num_queries, 1)
        )
        rois = torch.cat([roi_batch_idx, body_xyxy_feat.float()], dim=-1).reshape(-1, 5)
        roi_feat = roi_align(
            feat.float(),
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            aligned=True,
        ).to(feat.dtype)

        if self.head_context_fusion is not None:
            context = F.adaptive_avg_pool2d(feat, output_size=1)
            context = context.repeat_interleave(self.num_queries, dim=0)
            roi_feat = self.head_context_fusion(roi_feat, context)

        pred_head_rel = self.head_regressor(roi_feat.flatten(1)).sigmoid()
        pred_head_rel = pred_head_rel.view(batch_size, self.num_queries, 4)

        # Decode with clipped ROI geometry for consistency with roi_align inputs.
        body_xyxy_norm = body_xyxy_feat.clone()
        body_xyxy_norm[..., [0, 2]] /= float(feat_w)
        body_xyxy_norm[..., [1, 3]] /= float(feat_h)
        body_decode_boxes = xyxy_to_cxcywh(body_xyxy_norm.clamp(0.0, 1.0))
        pred_head_boxes = self._decode_head_relative(body_decode_boxes, pred_head_rel)

        return {
            "pred_logits": pred_logits,
            "pred_body_boxes": pred_body_boxes,
            "pred_head_boxes": pred_head_boxes,
        }


# Backward-compatible class alias.
CascadedBodyHeadDetector = CascadedBodyHeadDetectorSPPLite


@torch.no_grad()
def greedy_match_slots_split(
    pred_body: torch.Tensor,
    pred_head: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_body: torch.Tensor,
    gt_head: torch.Tensor,
    body_l1_w: float = 1.0,
    body_iou_w: float = 2.5,
    head_l1_w: float = 0.6,
    head_iou_w: float = 1.6,
    obj_w: float = 0.05,
    geo_w: float = 0.0,
    head_precision_hinge_w: float = 0.0,
    head_precision_thresh: float = 0.65,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split matching cost:
    - body side is weighted for stronger recall-oriented assignment
    - head side can add precision hinge for stricter localization
    """
    q_count = pred_body.shape[0]
    m_count = gt_body.shape[0]
    if q_count == 0 or m_count == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_body.device)
        return empty, empty

    body_l1 = torch.cdist(pred_body, gt_body, p=1)
    head_l1 = torch.cdist(pred_head, gt_head, p=1)

    iou_body = pairwise_iou_xyxy(cxcywh_to_xyxy(pred_body), cxcywh_to_xyxy(gt_body))
    iou_head = pairwise_iou_xyxy(cxcywh_to_xyxy(pred_head), cxcywh_to_xyxy(gt_head))

    cost = (
        body_l1_w * body_l1
        + body_iou_w * (1.0 - iou_body)
        + head_l1_w * head_l1
        + head_iou_w * (1.0 - iou_head)
    )

    obj_prob = pred_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    cost = cost + obj_w * (-torch.log(obj_prob))[:, None]

    if geo_w > 0.0:
        pred_geo = head_body_geometry(pred_body, pred_head)
        gt_geo = head_body_geometry(gt_body, gt_head)
        cost = cost + geo_w * torch.cdist(pred_geo, gt_geo, p=1)

    if head_precision_hinge_w > 0.0:
        hinge = (head_precision_thresh - iou_head).clamp(min=0.0)
        cost = cost + head_precision_hinge_w * hinge

    order = torch.argsort(cost.reshape(-1))
    used_q = torch.zeros(q_count, dtype=torch.bool, device=pred_body.device)
    used_m = torch.zeros(m_count, dtype=torch.bool, device=pred_body.device)

    matched_q = []
    matched_m = []
    for flat_idx in order:
        flat = int(flat_idx)
        q_idx = flat // m_count
        m_idx = flat % m_count
        if used_q[q_idx] or used_m[m_idx]:
            continue
        used_q[q_idx] = True
        used_m[m_idx] = True
        matched_q.append(q_idx)
        matched_m.append(m_idx)
        if len(matched_q) == min(q_count, m_count):
            break

    return (
        torch.tensor(matched_q, dtype=torch.long, device=pred_body.device),
        torch.tensor(matched_m, dtype=torch.long, device=pred_body.device),
    )


def body_head_set_loss_spp_lite(
    outputs,
    targets,
    w_obj: float = 1.0,
    w_body: float = 4.0,
    w_head: float = 20.0,
    w_iou: float = 2.0,
    w_geo: float = 0.0,
    obj_pos_weight: float = 30.0,
    box_smooth_l1_beta: float = 0.1,
    match_body_l1_w: float = 1.0,
    match_body_iou_w: float = 2.5,
    match_head_l1_w: float = 0.6,
    match_head_iou_w: float = 1.6,
    match_obj_w: float = 0.05,
    match_geo_w: float = 0.0,
    match_head_precision_hinge_w: float = 0.0,
    match_head_precision_thresh: float = 0.65,
    normalize_by: str = "matched",
    return_details: bool = False,
):
    pred_logits = outputs["pred_logits"]
    pred_body = outputs["pred_body_boxes"]
    pred_head = outputs["pred_head_boxes"]

    if normalize_by not in {"matched", "batch"}:
        raise ValueError("normalize_by must be 'matched' or 'batch'")
    if box_smooth_l1_beta <= 0:
        raise ValueError("box_smooth_l1_beta must be > 0")
    if obj_pos_weight <= 0:
        raise ValueError("obj_pos_weight must be > 0")

    batch_size, query_count = pred_logits.shape
    if batch_size == 0:
        raise ValueError("empty batch is not supported")
    if query_count < 1:
        raise ValueError("query_count must be >= 1")
    if len(targets) != batch_size:
        raise ValueError(f"targets length ({len(targets)}) must equal batch size ({batch_size})")

    total_obj = pred_logits.new_tensor(0.0)
    total_body = pred_logits.new_tensor(0.0)
    total_head = pred_logits.new_tensor(0.0)
    total_iou = pred_logits.new_tensor(0.0)
    total_geo = pred_logits.new_tensor(0.0)

    total_body_iou = pred_logits.new_tensor(0.0)
    total_head_iou = pred_logits.new_tensor(0.0)
    total_pos_obj = pred_logits.new_tensor(0.0)
    total_neg_obj = pred_logits.new_tensor(0.0)

    total_matched = 0
    total_neg = 0
    matched_per_image = []

    pos_weight = torch.tensor([obj_pos_weight], device=pred_logits.device, dtype=pred_logits.dtype)

    for b_idx in range(batch_size):
        if "body_boxes" not in targets[b_idx] or "head_boxes" not in targets[b_idx]:
            raise ValueError("each target must contain 'body_boxes' and 'head_boxes'")

        gt_body = _as_box_tensor(
            targets[b_idx]["body_boxes"],
            device=pred_body.device,
            dtype=pred_body.dtype,
        )
        gt_head = _as_box_tensor(
            targets[b_idx]["head_boxes"],
            device=pred_head.device,
            dtype=pred_head.dtype,
        )

        if gt_body.shape[0] != gt_head.shape[0]:
            raise ValueError("body_boxes and head_boxes must have the same number of boxes per image")
        if gt_body.shape[0] > query_count:
            raise ValueError(
                f"gt boxes per image ({gt_body.shape[0]}) exceed query_count ({query_count}); "
                "increase num_queries or cap per-image GT count"
            )

        matched_q, matched_m = greedy_match_slots_split(
            pred_body=pred_body[b_idx],
            pred_head=pred_head[b_idx],
            pred_logits=pred_logits[b_idx],
            gt_body=gt_body,
            gt_head=gt_head,
            body_l1_w=match_body_l1_w,
            body_iou_w=match_body_iou_w,
            head_l1_w=match_head_l1_w,
            head_iou_w=match_head_iou_w,
            obj_w=match_obj_w,
            geo_w=match_geo_w,
            head_precision_hinge_w=match_head_precision_hinge_w,
            head_precision_thresh=match_head_precision_thresh,
        )

        matched_count = int(matched_q.numel())
        matched_per_image.append(matched_count)
        total_matched += matched_count

        obj_target = torch.zeros(query_count, device=pred_logits.device, dtype=pred_logits.dtype)
        if matched_count > 0:
            obj_target[matched_q] = 1.0
        total_obj = total_obj + F.binary_cross_entropy_with_logits(
            pred_logits[b_idx],
            obj_target,
            pos_weight=pos_weight,
        )

        obj_prob = pred_logits[b_idx].sigmoid()
        if matched_count > 0:
            total_pos_obj = total_pos_obj + obj_prob[matched_q].sum()
        neg_mask = torch.ones(query_count, dtype=torch.bool, device=pred_logits.device)
        if matched_count > 0:
            neg_mask[matched_q] = False
        neg_count = int(neg_mask.sum().item())
        total_neg += neg_count
        if neg_count > 0:
            total_neg_obj = total_neg_obj + obj_prob[neg_mask].sum()

        if matched_count == 0:
            continue

        pb = pred_body[b_idx][matched_q]
        ph = pred_head[b_idx][matched_q]
        gb = gt_body[matched_m]
        gh = gt_head[matched_m]

        total_body = total_body + F.smooth_l1_loss(pb, gb, beta=box_smooth_l1_beta, reduction="sum")
        total_head = total_head + F.smooth_l1_loss(ph, gh, beta=box_smooth_l1_beta, reduction="sum")

        body_iou = pairwise_iou_xyxy(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb)).diag()
        head_iou = pairwise_iou_xyxy(cxcywh_to_xyxy(ph), cxcywh_to_xyxy(gh)).diag()
        total_iou = total_iou + (1.0 - body_iou).sum() + (1.0 - head_iou).sum()
        total_body_iou = total_body_iou + body_iou.sum()
        total_head_iou = total_head_iou + head_iou.sum()

        if w_geo > 0.0:
            pred_geo = head_body_geometry(pb, ph)
            gt_geo = head_body_geometry(gb, gh)
            total_geo = total_geo + F.smooth_l1_loss(
                pred_geo,
                gt_geo,
                beta=box_smooth_l1_beta,
                reduction="sum",
            )

    obj_loss = total_obj / batch_size
    if normalize_by == "batch":
        body_loss = total_body / batch_size
        head_loss = total_head / batch_size
        iou_loss = total_iou / batch_size
        geo_loss = total_geo / batch_size
    else:
        norm = max(total_matched, 1)
        body_loss = total_body / norm
        head_loss = total_head / norm
        iou_loss = total_iou / (2 * norm)
        geo_loss = total_geo / norm

    total_loss = w_obj * obj_loss + w_body * body_loss + w_head * head_loss + w_iou * iou_loss + w_geo * geo_loss

    if not return_details:
        return total_loss

    matched_norm = max(total_matched, 1)
    details = {
        "total": total_loss,
        "obj": obj_loss,
        "body": body_loss,
        "head": head_loss,
        "iou": iou_loss,
        "geo": geo_loss,
        "num_matched": total_matched,
        "matched_ratio": total_matched / float(batch_size * query_count),
        "matched_per_image": matched_per_image,
        "mean_body_iou": (total_body_iou / matched_norm).detach(),
        "mean_head_iou": (total_head_iou / matched_norm).detach(),
        "mean_pos_obj_prob": (total_pos_obj / matched_norm).detach(),
        "mean_neg_obj_prob": (total_neg_obj / max(total_neg, 1)).detach(),
        "weights": {
            "w_obj": w_obj,
            "w_body": w_body,
            "w_head": w_head,
            "w_iou": w_iou,
            "w_geo": w_geo,
        },
    }
    return details


def scheduled_body_head_weights(
    step: int,
    total_steps: int,
    w_body_start: float = 8.0,
    w_body_end: float = 4.0,
    w_head_start: float = 10.0,
    w_head_end: float = 20.0,
    curriculum_power: float = 1.0,
) -> Tuple[float, float, float]:
    if curriculum_power <= 0:
        raise ValueError("curriculum_power must be > 0")

    if total_steps <= 0:
        progress = 1.0
    else:
        progress = float(min(max(step, 0), total_steps)) / float(total_steps)
    progress = progress ** curriculum_power

    w_body = w_body_start + (w_body_end - w_body_start) * progress
    w_head = w_head_start + (w_head_end - w_head_start) * progress
    return w_body, w_head, progress


def body_head_set_loss_with_curriculum(
    outputs,
    targets,
    step: int,
    total_steps: int,
    w_obj: float = 1.0,
    w_iou: float = 2.0,
    w_geo: float = 0.0,
    w_body_start: float = 8.0,
    w_body_end: float = 4.0,
    w_head_start: float = 10.0,
    w_head_end: float = 20.0,
    curriculum_power: float = 1.0,
    return_details: bool = False,
    **loss_kwargs,
):
    w_body, w_head, progress = scheduled_body_head_weights(
        step=step,
        total_steps=total_steps,
        w_body_start=w_body_start,
        w_body_end=w_body_end,
        w_head_start=w_head_start,
        w_head_end=w_head_end,
        curriculum_power=curriculum_power,
    )

    details = body_head_set_loss_spp_lite(
        outputs,
        targets,
        w_obj=w_obj,
        w_body=w_body,
        w_head=w_head,
        w_iou=w_iou,
        w_geo=w_geo,
        return_details=True,
        **loss_kwargs,
    )

    if not return_details:
        return details["total"]

    details["curriculum"] = {
        "step": int(step),
        "total_steps": int(total_steps),
        "progress": float(progress),
        "w_body": float(w_body),
        "w_head": float(w_head),
    }
    return details


def cascaded_body_head_loss_spp_lite(outputs, targets, **kwargs):
    return body_head_set_loss_spp_lite(outputs, targets, **kwargs)


# Backward-compatible loss alias.
cascaded_body_head_loss = cascaded_body_head_loss_spp_lite


@torch.no_grad()
def infer_slots(outputs, conf_thresh: float = 0.5):
    pred_logits = outputs["pred_logits"]
    pred_body = outputs["pred_body_boxes"]
    pred_head = outputs["pred_head_boxes"]

    batch_size = pred_logits.shape[0]
    result = []
    for b_idx in range(batch_size):
        conf = pred_logits[b_idx].sigmoid()
        keep = conf > conf_thresh
        result.append(
            {
                "conf": conf[keep],
                "body_boxes_xyxy": cxcywh_to_xyxy(pred_body[b_idx][keep]).clamp(0.0, 1.0),
                "head_boxes_xyxy": cxcywh_to_xyxy(pred_head[b_idx][keep]).clamp(0.0, 1.0),
            }
        )
    return result


if __name__ == "__main__":
    torch.manual_seed(0)

    model = CascadedBodyHeadDetectorSPPLite(
        num_queries=10,
        backbone_type="tiny48",
        roi_size=5,
        use_tiny48_cff=True,
    )

    x = torch.randn(2, 3, 256, 256)
    outputs = model(x)
    print("outputs:", {k: tuple(v.shape) for k, v in outputs.items()})

    targets = [
        {
            "body_boxes": torch.tensor([[0.50, 0.55, 0.38, 0.62]], dtype=torch.float32),
            "head_boxes": torch.tensor([[0.50, 0.31, 0.14, 0.15]], dtype=torch.float32),
        },
        {
            "body_boxes": torch.tensor([[0.65, 0.58, 0.30, 0.52]], dtype=torch.float32),
            "head_boxes": torch.tensor([[0.64, 0.34, 0.12, 0.13]], dtype=torch.float32),
        },
    ]

    loss_dict = body_head_set_loss_with_curriculum(
        outputs,
        targets,
        step=10,
        total_steps=100,
        return_details=True,
    )
    print("loss:", float(loss_dict["total"]))
    print("curriculum:", loss_dict["curriculum"])
