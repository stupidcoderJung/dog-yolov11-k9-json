import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def pairwise_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (Na, 4) xyxy
    b: (Nb, 4) xyxy
    returns: (Na, Nb)
    """
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


def cxcywh_stats(boxes: torch.Tensor):
    """
    Simple sanity stats for normalized cxcywh boxes.
    """
    if boxes.numel() == 0:
        return {
            "min": 0.0,
            "max": 0.0,
            "in_01_ratio": 1.0,
            "positive_wh_ratio": 1.0,
            "valid_xyxy_ratio": 1.0,
        }

    xyxy = cxcywh_to_xyxy(boxes)
    x1, y1, x2, y2 = xyxy.unbind(-1)
    in_01_ratio = ((boxes >= 0.0) & (boxes <= 1.0)).all(dim=-1).float().mean()
    positive_wh_ratio = ((boxes[..., 2] > 0.0) & (boxes[..., 3] > 0.0)).float().mean()
    valid_xyxy_ratio = ((x2 > x1) & (y2 > y1)).float().mean()
    return {
        "min": float(boxes.min().item()),
        "max": float(boxes.max().item()),
        "in_01_ratio": float(in_01_ratio.item()),
        "positive_wh_ratio": float(positive_wh_ratio.item()),
        "valid_xyxy_ratio": float(valid_xyxy_ratio.item()),
    }


@torch.no_grad()
def greedy_match_slots(
    pred_body: torch.Tensor,
    pred_head: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_body: torch.Tensor,
    gt_head: torch.Tensor,
    w_l1: float = 1.0,
    w_iou: float = 2.0,
    w_obj: float = 0.05,
):
    """
    pred_*: (Q, 4), pred_logits: (Q,), gt_*: (M, 4)
    returns: matched_pred_idx, matched_gt_idx

    Note:
      This is a greedy approximation, intentionally used to keep this study
      implementation simple. It is not globally optimal like Hungarian matching.
    """
    q_count = pred_body.shape[0]
    m_count = gt_body.shape[0]
    if q_count == 0 or m_count == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_body.device)
        return empty, empty

    l1 = torch.cdist(pred_body, gt_body, p=1) + torch.cdist(pred_head, gt_head, p=1)
    iou_body = pairwise_iou_xyxy(cxcywh_to_xyxy(pred_body), cxcywh_to_xyxy(gt_body))
    iou_head = pairwise_iou_xyxy(cxcywh_to_xyxy(pred_head), cxcywh_to_xyxy(gt_head))

    cost = w_l1 * l1 + w_iou * ((1.0 - iou_body) + (1.0 - iou_head))
    obj = pred_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    cost = cost + (-torch.log(obj))[:, None] * w_obj

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


class BodyHeadSlotDetector(nn.Module):
    """
    Minimal slot detector without transformer.

    Per slot output:
      - objectness logit
      - body box (cx, cy, w, h), normalized
      - head box (cx, cy, w, h), normalized

    Design (simple, split body/head paths):
      1) shared image backbone -> body logits/boxes
      2) per-body ROI features -> tiny head backbone -> head box
    """

    def __init__(self, num_queries: int = 50, roi_size: int = 5):
        super().__init__()
        if num_queries < 1:
            raise ValueError("num_queries must be >= 1")
        if roi_size < 2:
            raise ValueError("roi_size must be >= 2")
        self.num_queries = num_queries
        self.roi_size = roi_size

        # Shared image backbone (for body branch and ROI extraction).
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
        )

        # Stage 1: body/objectness from global pooled feature.
        self.body_objectness_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, num_queries),
        )
        self.body_box_head = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, num_queries * 4),
        )

        # Stage 2: tiny head-only backbone on body ROI features.
        self.head_roi_backbone = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(True),
        )
        self.head_box_head = nn.Sequential(
            nn.Linear(32 * roi_size * roi_size, 128),
            nn.ReLU(True),
            nn.Linear(128, 4),  # relative cxcywh in body ROI
        )

    @staticmethod
    def _decode_head_relative(body_boxes: torch.Tensor, head_relative: torch.Tensor) -> torch.Tensor:
        """
        body_boxes: (B, Q, 4) normalized cxcywh
        head_relative: (B, Q, 4) normalized relative cxcywh in body ROI
        returns absolute normalized cxcywh
        """
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

    def forward(self, x: torch.Tensor):
        bsz = x.shape[0]
        feat = self.backbone(x)
        pooled = feat.mean(dim=(2, 3))

        # Stage 1: body prediction.
        pred_logits = self.body_objectness_head(pooled)
        pred_body_boxes = self.body_box_head(pooled).view(bsz, self.num_queries, 4).sigmoid()

        # Stage 2: head prediction from body ROI features.
        feat_h, feat_w = feat.shape[2], feat.shape[3]
        body_xyxy = cxcywh_to_xyxy(pred_body_boxes)
        body_xyxy_feat = body_xyxy.clone()
        body_xyxy_feat[..., [0, 2]] *= float(feat_w)
        body_xyxy_feat[..., [1, 3]] *= float(feat_h)
        body_xyxy_feat[..., [0, 2]] = body_xyxy_feat[..., [0, 2]].clamp(0.0, float(feat_w))
        body_xyxy_feat[..., [1, 3]] = body_xyxy_feat[..., [1, 3]].clamp(0.0, float(feat_h))

        roi_batch_idx = (
            torch.arange(bsz, device=x.device, dtype=torch.float32)
            .view(bsz, 1, 1)
            .expand(bsz, self.num_queries, 1)
        )
        rois = torch.cat([roi_batch_idx, body_xyxy_feat.float()], dim=-1).reshape(-1, 5)
        roi_feat = roi_align(
            feat.float(),
            rois,
            output_size=(self.roi_size, self.roi_size),
            spatial_scale=1.0,
            aligned=True,
        ).to(feat.dtype)
        head_feat = self.head_roi_backbone(roi_feat)
        pred_head_rel = self.head_box_head(head_feat.flatten(1)).sigmoid().view(bsz, self.num_queries, 4)
        pred_head_boxes = self._decode_head_relative(pred_body_boxes, pred_head_rel)

        return {
            "pred_logits": pred_logits,
            "pred_body_boxes": pred_body_boxes,
            "pred_head_boxes": pred_head_boxes,
        }


def body_head_set_loss(
    outputs,
    targets,
    w_obj: float = 1.0,
    w_body: float = 4.0,
    w_head: float = 4.0,
    w_iou: float = 2.0,
    obj_pos_weight: float = 1.0,
    box_smooth_l1_beta: float = 0.1,
    match_w_l1: float = 1.0,
    match_w_iou: float = 2.0,
    match_w_obj: float = 0.05,
    normalize_by: str = "matched",
    validate_targets: bool = False,
    strict_target_check: bool = False,
    return_details: bool = False,
    debug: bool = False,
):
    """
    targets: list of len B
      each item:
        {
          "body_boxes": (Mi, 4) normalized cxcywh
          "head_boxes": (Mi, 4) normalized cxcywh
        }
    """
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
    if query_count == 0:
        raise ValueError("query_count must be >= 1")
    if len(targets) != batch_size:
        raise ValueError(f"targets length ({len(targets)}) must equal batch size ({batch_size})")

    total_obj = pred_logits.new_tensor(0.0)
    total_body = pred_logits.new_tensor(0.0)
    total_head = pred_logits.new_tensor(0.0)
    total_iou = pred_logits.new_tensor(0.0)

    total_matched = 0
    total_neg = 0
    total_body_abs = pred_logits.new_tensor(0.0)
    total_head_abs = pred_logits.new_tensor(0.0)
    total_body_iou = pred_logits.new_tensor(0.0)
    total_head_iou = pred_logits.new_tensor(0.0)
    total_pos_obj = pred_logits.new_tensor(0.0)
    total_neg_obj = pred_logits.new_tensor(0.0)

    checked_images = 0
    sum_body_in01 = 0.0
    sum_head_in01 = 0.0
    sum_body_valid = 0.0
    sum_head_valid = 0.0

    matched_per_image = []
    pos_weight = torch.tensor([obj_pos_weight], device=pred_logits.device, dtype=pred_logits.dtype)

    for b_idx in range(batch_size):
        gt_body = targets[b_idx]["body_boxes"].to(device=pred_body.device, dtype=pred_body.dtype)
        gt_head = targets[b_idx]["head_boxes"].to(device=pred_head.device, dtype=pred_head.dtype)
        if gt_body.shape[0] != gt_head.shape[0]:
            raise ValueError("body_boxes and head_boxes must have the same number of boxes per image")
        if gt_body.shape[0] > query_count:
            raise ValueError(
                f"gt boxes per image ({gt_body.shape[0]}) exceed query_count ({query_count}); "
                "increase num_queries or cap per-image GT count"
            )

        if validate_targets or strict_target_check or debug:
            body_stats = cxcywh_stats(gt_body)
            head_stats = cxcywh_stats(gt_head)

            if validate_targets:
                checked_images += 1
                sum_body_in01 += body_stats["in_01_ratio"]
                sum_head_in01 += head_stats["in_01_ratio"]
                sum_body_valid += body_stats["valid_xyxy_ratio"]
                sum_head_valid += head_stats["valid_xyxy_ratio"]

            if strict_target_check:
                bad_body = (
                    body_stats["in_01_ratio"] < 1.0
                    or body_stats["positive_wh_ratio"] < 1.0
                    or body_stats["valid_xyxy_ratio"] < 1.0
                )
                bad_head = (
                    head_stats["in_01_ratio"] < 1.0
                    or head_stats["positive_wh_ratio"] < 1.0
                    or head_stats["valid_xyxy_ratio"] < 1.0
                )
                if bad_body or bad_head:
                    raise ValueError(
                        f"invalid GT at image {b_idx}: "
                        f"body={body_stats}, head={head_stats}"
                    )

            if debug:
                print(
                    f"[debug gt] img={b_idx} body_in01={body_stats['in_01_ratio']:.3f} "
                    f"head_in01={head_stats['in_01_ratio']:.3f} "
                    f"body_valid={body_stats['valid_xyxy_ratio']:.3f} "
                    f"head_valid={head_stats['valid_xyxy_ratio']:.3f}"
                )

        matched_q, matched_m = greedy_match_slots(
            pred_body[b_idx],
            pred_head[b_idx],
            pred_logits[b_idx],
            gt_body,
            gt_head,
            w_l1=match_w_l1,
            w_iou=match_w_iou,
            w_obj=match_w_obj,
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
            if debug:
                print(f"[debug match] img={b_idx} GT={gt_body.shape[0]} matched=0/{query_count}")
            continue

        pb = pred_body[b_idx][matched_q]
        ph = pred_head[b_idx][matched_q]
        gb = gt_body[matched_m]
        gh = gt_head[matched_m]

        total_body = total_body + F.smooth_l1_loss(
            pb,
            gb,
            beta=box_smooth_l1_beta,
            reduction="sum",
        )
        total_head = total_head + F.smooth_l1_loss(
            ph,
            gh,
            beta=box_smooth_l1_beta,
            reduction="sum",
        )

        total_body_abs = total_body_abs + (pb - gb).abs().sum()
        total_head_abs = total_head_abs + (ph - gh).abs().sum()

        body_iou = pairwise_iou_xyxy(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb)).diag()
        head_iou = pairwise_iou_xyxy(cxcywh_to_xyxy(ph), cxcywh_to_xyxy(gh)).diag()
        total_iou = total_iou + (1.0 - body_iou).sum() + (1.0 - head_iou).sum()
        total_body_iou = total_body_iou + body_iou.sum()
        total_head_iou = total_head_iou + head_iou.sum()

        if debug:
            print(
                f"[debug match] img={b_idx} GT={gt_body.shape[0]} matched={matched_count}/{query_count} "
                f"body_abs={(pb - gb).abs().mean().item():.4f} "
                f"head_abs={(ph - gh).abs().mean().item():.4f}"
            )

    obj_loss = total_obj / batch_size
    if normalize_by == "batch":
        body_loss = total_body / batch_size
        head_loss = total_head / batch_size
        iou_loss = total_iou / batch_size
    else:
        norm = max(total_matched, 1)
        body_loss = total_body / norm
        head_loss = total_head / norm
        iou_loss = total_iou / (2 * norm)

    total_loss = w_obj * obj_loss + w_body * body_loss + w_head * head_loss + w_iou * iou_loss
    if not return_details:
        return total_loss

    matched_norm = max(total_matched, 1)
    coord_norm = max(total_matched * 4, 1)
    details = {
        "total": total_loss,
        "obj": obj_loss,
        "body": body_loss,
        "head": head_loss,
        "iou": iou_loss,
        "num_matched": total_matched,
        "matched_ratio": total_matched / float(batch_size * query_count),
        "matched_per_image": matched_per_image,
        "mean_abs_body_coord_err": (total_body_abs / coord_norm).detach(),
        "mean_abs_head_coord_err": (total_head_abs / coord_norm).detach(),
        "mean_body_iou": (total_body_iou / matched_norm).detach(),
        "mean_head_iou": (total_head_iou / matched_norm).detach(),
        "mean_pos_obj_prob": (total_pos_obj / matched_norm).detach(),
        "mean_neg_obj_prob": (total_neg_obj / max(total_neg, 1)).detach(),
    }
    if validate_targets and checked_images > 0:
        details["target_stats"] = {
            "checked_images": checked_images,
            "mean_body_in_01_ratio": sum_body_in01 / checked_images,
            "mean_head_in_01_ratio": sum_head_in01 / checked_images,
            "mean_body_valid_xyxy_ratio": sum_body_valid / checked_images,
            "mean_head_valid_xyxy_ratio": sum_head_valid / checked_images,
        }
    return details


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
                "body_boxes_xyxy": cxcywh_to_xyxy(pred_body[b_idx][keep]).clamp(0, 1),
                "head_boxes_xyxy": cxcywh_to_xyxy(pred_head[b_idx][keep]).clamp(0, 1),
            }
        )
    return result


if __name__ == "__main__":
    torch.manual_seed(0)
    model = BodyHeadSlotDetector(num_queries=40)
    x = torch.randn(2, 3, 256, 256)
    targets = [
        {
            "body_boxes": torch.tensor([[0.30, 0.35, 0.24, 0.46], [0.74, 0.42, 0.20, 0.42]], dtype=torch.float32),
            "head_boxes": torch.tensor([[0.31, 0.21, 0.12, 0.14], [0.73, 0.25, 0.10, 0.12]], dtype=torch.float32),
        },
        {
            "body_boxes": torch.tensor([[0.50, 0.55, 0.30, 0.50]], dtype=torch.float32),
            "head_boxes": torch.tensor([[0.51, 0.33, 0.13, 0.15]], dtype=torch.float32),
        },
    ]

    outputs = model(x)
    loss_dict = body_head_set_loss(
        outputs,
        targets,
        return_details=True,
        validate_targets=True,
        debug=True,
    )
    loss = loss_dict["total"]
    loss.backward()
    print("loss:", loss.item())
    print(
        "parts:",
        {
            "obj": float(loss_dict["obj"]),
            "body": float(loss_dict["body"]),
            "head": float(loss_dict["head"]),
            "iou": float(loss_dict["iou"]),
            "matched": loss_dict["num_matched"],
            "matched_ratio": loss_dict["matched_ratio"],
        },
    )
    print("target_stats:", loss_dict.get("target_stats"))
    pred = infer_slots(outputs, conf_thresh=0.3)
    print("pred counts:", [p["body_boxes_xyxy"].shape[0] for p in pred])
