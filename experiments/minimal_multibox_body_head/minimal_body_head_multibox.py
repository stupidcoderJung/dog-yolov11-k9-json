import torch
import torch.nn as nn
import torch.nn.functional as F


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)


def xyxy_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    a: (Na, 4), b: (Nb, 4) in xyxy format
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


@torch.no_grad()
def greedy_match(
    pred_body_boxes: torch.Tensor,
    pred_head_boxes: torch.Tensor,
    pred_logits: torch.Tensor,
    gt_body_boxes: torch.Tensor,
    gt_head_boxes: torch.Tensor,
    w_l1: float = 1.0,
    w_iou: float = 2.0,
    w_obj: float = 0.05,
):
    """
    Input shapes:
      pred_body_boxes: (Q, 4)
      pred_head_boxes: (Q, 4)
      pred_logits:     (Q,)
      gt_body_boxes:   (M, 4)
      gt_head_boxes:   (M, 4)
    """
    q_count = pred_body_boxes.shape[0]
    m_count = gt_body_boxes.shape[0]
    if q_count == 0 or m_count == 0:
        empty = torch.empty(0, dtype=torch.long, device=pred_body_boxes.device)
        return empty, empty

    l1_body = torch.cdist(pred_body_boxes, gt_body_boxes, p=1)
    l1_head = torch.cdist(pred_head_boxes, gt_head_boxes, p=1)

    iou_body = xyxy_iou(cxcywh_to_xyxy(pred_body_boxes), cxcywh_to_xyxy(gt_body_boxes))
    iou_head = xyxy_iou(cxcywh_to_xyxy(pred_head_boxes), cxcywh_to_xyxy(gt_head_boxes))

    cost = w_l1 * (l1_body + l1_head) + w_iou * ((1.0 - iou_body) + (1.0 - iou_head))

    obj = pred_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    cost = cost + (-torch.log(obj))[:, None] * w_obj

    order = torch.argsort(cost.reshape(-1))
    used_q = torch.zeros(q_count, dtype=torch.bool, device=pred_body_boxes.device)
    used_m = torch.zeros(m_count, dtype=torch.bool, device=pred_body_boxes.device)

    matched_q, matched_m = [], []
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
        torch.tensor(matched_q, dtype=torch.long, device=pred_body_boxes.device),
        torch.tensor(matched_m, dtype=torch.long, device=pred_body_boxes.device),
    )


class MinimalBodyHeadMultiBox(nn.Module):
    """
    Minimal slot-based detector without transformer.

    Output per query:
      - objectness logit (1)
      - body box (cx, cy, w, h)
      - head box (cx, cy, w, h)
    """

    def __init__(self, num_queries: int = 50):
        super().__init__()
        self.num_queries = num_queries

        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(True),
        )

        self.fc = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, self.num_queries * 9),
        )

    def forward(self, x: torch.Tensor):
        batch_size = x.shape[0]
        feat = self.backbone(x)
        pooled = feat.mean(dim=(2, 3))

        out = self.fc(pooled).view(batch_size, self.num_queries, 9)
        pred_logits = out[..., 0]
        pred_body_boxes = out[..., 1:5].sigmoid()
        pred_head_boxes = out[..., 5:9].sigmoid()
        return {
            "pred_logits": pred_logits,
            "pred_body_boxes": pred_body_boxes,
            "pred_head_boxes": pred_head_boxes,
        }


def multibox_body_head_loss(
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
    return_details: bool = False,
    debug: bool = False,
):
    """
    targets: list of dict, len B
      each target:
        {
          "body_boxes": (Mi, 4) normalized cxcywh,
          "head_boxes": (Mi, 4) normalized cxcywh
        }
    """
    pred_logits = outputs["pred_logits"]
    pred_body_boxes = outputs["pred_body_boxes"]
    pred_head_boxes = outputs["pred_head_boxes"]
    if box_smooth_l1_beta <= 0:
        raise ValueError("box_smooth_l1_beta must be > 0")
    if normalize_by not in {"matched", "batch"}:
        raise ValueError("normalize_by must be 'matched' or 'batch'")

    batch_size, query_count = pred_logits.shape
    if batch_size == 0:
        raise ValueError("empty batch is not supported")
    if len(targets) != batch_size:
        raise ValueError(f"targets length ({len(targets)}) must equal batch size ({batch_size})")

    total_obj = pred_logits.new_tensor(0.0)
    total_body_sum = pred_logits.new_tensor(0.0)
    total_head_sum = pred_logits.new_tensor(0.0)
    total_iou_sum = pred_logits.new_tensor(0.0)
    total_matched = 0
    total_body_abs = pred_logits.new_tensor(0.0)
    total_head_abs = pred_logits.new_tensor(0.0)
    total_body_iou = pred_logits.new_tensor(0.0)
    total_head_iou = pred_logits.new_tensor(0.0)
    total_pos_obj = pred_logits.new_tensor(0.0)
    total_neg_obj = pred_logits.new_tensor(0.0)
    total_neg = 0
    matched_per_image = []

    for batch_idx in range(batch_size):
        gt_body = targets[batch_idx]["body_boxes"].to(
            device=pred_body_boxes.device,
            dtype=pred_body_boxes.dtype,
        )
        gt_head = targets[batch_idx]["head_boxes"].to(
            device=pred_head_boxes.device,
            dtype=pred_head_boxes.dtype,
        )
        if gt_body.shape[0] != gt_head.shape[0]:
            raise ValueError("body_boxes and head_boxes must have the same number of boxes per image")

        matched_q, matched_m = greedy_match(
            pred_body_boxes[batch_idx],
            pred_head_boxes[batch_idx],
            pred_logits[batch_idx],
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
        if matched_q.numel() > 0:
            obj_target[matched_q] = 1.0
        pos_weight = torch.tensor([obj_pos_weight], device=pred_logits.device, dtype=pred_logits.dtype)
        total_obj = total_obj + F.binary_cross_entropy_with_logits(
            pred_logits[batch_idx],
            obj_target,
            pos_weight=pos_weight,
        )

        obj_prob = pred_logits[batch_idx].sigmoid()
        if matched_q.numel() > 0:
            total_pos_obj = total_pos_obj + obj_prob[matched_q].sum()
        neg_mask = torch.ones(query_count, dtype=torch.bool, device=pred_logits.device)
        if matched_q.numel() > 0:
            neg_mask[matched_q] = False
        neg_count = int(neg_mask.sum().item())
        total_neg += neg_count
        if neg_count > 0:
            total_neg_obj = total_neg_obj + obj_prob[neg_mask].sum()

        if matched_q.numel() > 0:
            pb = pred_body_boxes[batch_idx][matched_q]
            ph = pred_head_boxes[batch_idx][matched_q]
            gb = gt_body[matched_m]
            gh = gt_head[matched_m]

            # Smaller beta increases gradient magnitude for normalized [0, 1] box coordinates.
            total_body_sum = total_body_sum + F.smooth_l1_loss(
                pb,
                gb,
                beta=box_smooth_l1_beta,
                reduction="sum",
            )
            total_head_sum = total_head_sum + F.smooth_l1_loss(
                ph,
                gh,
                beta=box_smooth_l1_beta,
                reduction="sum",
            )
            total_body_abs = total_body_abs + (pb - gb).abs().sum()
            total_head_abs = total_head_abs + (ph - gh).abs().sum()

            iou_body = xyxy_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb)).diag()
            iou_head = xyxy_iou(cxcywh_to_xyxy(ph), cxcywh_to_xyxy(gh)).diag()
            total_iou_sum = total_iou_sum + (1.0 - iou_body).sum() + (1.0 - iou_head).sum()
            total_body_iou = total_body_iou + iou_body.sum()
            total_head_iou = total_head_iou + iou_head.sum()

            if debug:
                body_abs = (pb - gb).abs().mean().item()
                head_abs = (ph - gh).abs().mean().item()
                body_iou = iou_body.mean().item()
                head_iou = iou_head.mean().item()
                print(
                    f"[multibox debug] img={batch_idx} matched={matched_count}/{query_count} "
                    f"body_abs_mean={body_abs:.4f} head_abs_mean={head_abs:.4f} "
                    f"body_iou_mean={body_iou:.4f} head_iou_mean={head_iou:.4f}"
                )
        elif debug:
            print(f"[multibox debug] img={batch_idx} matched=0/{query_count}")

    obj_loss = total_obj / batch_size
    if normalize_by == "batch":
        body_loss = total_body_sum / batch_size
        head_loss = total_head_sum / batch_size
        iou_loss = total_iou_sum / batch_size
    else:
        norm = max(total_matched, 1)
        body_loss = total_body_sum / norm
        head_loss = total_head_sum / norm
        # (body + head) IoU penalties are accumulated, so divide by 2 * matched.
        iou_loss = total_iou_sum / (2 * norm)

    total_loss = w_obj * obj_loss + w_body * body_loss + w_head * head_loss + w_iou * iou_loss
    if return_details:
        matched_norm = max(total_matched, 1)
        coord_norm = max(total_matched * 4, 1)
        return {
            "total": total_loss,
            "obj": obj_loss,
            "body": body_loss,
            "head": head_loss,
            "iou": iou_loss,
            "num_matched": total_matched,
            "matched_per_image": matched_per_image,
            "matched_ratio": total_matched / float(batch_size * query_count),
            "mean_abs_body_coord_err": (total_body_abs / coord_norm).detach(),
            "mean_abs_head_coord_err": (total_head_abs / coord_norm).detach(),
            "mean_body_iou": (total_body_iou / matched_norm).detach(),
            "mean_head_iou": (total_head_iou / matched_norm).detach(),
            "mean_pos_obj_prob": (total_pos_obj / matched_norm).detach(),
            "mean_neg_obj_prob": (total_neg_obj / max(total_neg, 1)).detach(),
            "box_smooth_l1_beta": float(box_smooth_l1_beta),
            "matcher_weights": {
                "l1": match_w_l1,
                "iou": match_w_iou,
                "obj": match_w_obj,
            },
        }
    return total_loss


@torch.no_grad()
def infer(outputs, conf_thresh: float = 0.5):
    pred_logits = outputs["pred_logits"]
    pred_body_boxes = outputs["pred_body_boxes"]
    pred_head_boxes = outputs["pred_head_boxes"]
    batch_size = pred_logits.shape[0]

    results = []
    for batch_idx in range(batch_size):
        conf = pred_logits[batch_idx].sigmoid()
        keep = conf > conf_thresh
        body_xyxy = cxcywh_to_xyxy(pred_body_boxes[batch_idx][keep]).clamp(0, 1)
        head_xyxy = cxcywh_to_xyxy(pred_head_boxes[batch_idx][keep]).clamp(0, 1)
        results.append(
            {
                "body_boxes_xyxy": body_xyxy,
                "head_boxes_xyxy": head_xyxy,
                "conf": conf[keep],
            }
        )
    return results


if __name__ == "__main__":
    torch.manual_seed(0)
    model = MinimalBodyHeadMultiBox(num_queries=40)

    x = torch.randn(2, 3, 256, 256)
    targets = [
        {
            "body_boxes": torch.tensor(
                [
                    [0.30, 0.35, 0.26, 0.48],
                    [0.74, 0.42, 0.20, 0.42],
                ],
                dtype=torch.float32,
            ),
            "head_boxes": torch.tensor(
                [
                    [0.31, 0.21, 0.12, 0.14],
                    [0.73, 0.25, 0.10, 0.12],
                ],
                dtype=torch.float32,
            ),
        },
        {
            "body_boxes": torch.tensor([[0.50, 0.55, 0.30, 0.50]], dtype=torch.float32),
            "head_boxes": torch.tensor([[0.51, 0.33, 0.13, 0.15]], dtype=torch.float32),
        },
    ]

    outputs = model(x)
    loss_dict = multibox_body_head_loss(outputs, targets, return_details=True, debug=True)
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
            "body_abs": float(loss_dict["mean_abs_body_coord_err"]),
            "head_abs": float(loss_dict["mean_abs_head_coord_err"]),
            "body_iou": float(loss_dict["mean_body_iou"]),
            "head_iou": float(loss_dict["mean_head_iou"]),
            "pos_obj": float(loss_dict["mean_pos_obj_prob"]),
            "neg_obj": float(loss_dict["mean_neg_obj_prob"]),
        },
    )

    preds = infer(outputs, conf_thresh=0.3)
    print("pred counts:", [pred["body_boxes_xyxy"].shape[0] for pred in preds])
