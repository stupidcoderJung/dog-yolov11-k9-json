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

    cost = (l1_body + l1_head) + ((1.0 - iou_body) + (1.0 - iou_head))

    obj = pred_logits.sigmoid().clamp(1e-6, 1.0 - 1e-6)
    cost = cost + (-torch.log(obj))[:, None] * 0.25

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

    batch_size, query_count = pred_logits.shape

    total_obj = 0.0
    total_body = 0.0
    total_head = 0.0
    total_iou = 0.0

    for batch_idx in range(batch_size):
        gt_body = targets[batch_idx]["body_boxes"].to(pred_body_boxes.device)
        gt_head = targets[batch_idx]["head_boxes"].to(pred_head_boxes.device)
        if gt_body.shape[0] != gt_head.shape[0]:
            raise ValueError("body_boxes and head_boxes must have the same number of boxes per image")

        matched_q, matched_m = greedy_match(
            pred_body_boxes[batch_idx],
            pred_head_boxes[batch_idx],
            pred_logits[batch_idx],
            gt_body,
            gt_head,
        )

        obj_target = torch.zeros(query_count, device=pred_logits.device, dtype=pred_logits.dtype)
        if matched_q.numel() > 0:
            obj_target[matched_q] = 1.0
        total_obj = total_obj + F.binary_cross_entropy_with_logits(pred_logits[batch_idx], obj_target)

        if matched_q.numel() > 0:
            pb = pred_body_boxes[batch_idx][matched_q]
            ph = pred_head_boxes[batch_idx][matched_q]
            gb = gt_body[matched_m]
            gh = gt_head[matched_m]

            total_body = total_body + F.smooth_l1_loss(pb, gb)
            total_head = total_head + F.smooth_l1_loss(ph, gh)

            iou_body = xyxy_iou(cxcywh_to_xyxy(pb), cxcywh_to_xyxy(gb)).diag()
            iou_head = xyxy_iou(cxcywh_to_xyxy(ph), cxcywh_to_xyxy(gh)).diag()
            total_iou = total_iou + (1.0 - 0.5 * (iou_body + iou_head)).mean()

    loss = (w_obj * total_obj + w_body * total_body + w_head * total_head + w_iou * total_iou) / batch_size
    return loss


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
    loss = multibox_body_head_loss(outputs, targets)
    loss.backward()
    print("loss:", loss.item())

    preds = infer(outputs, conf_thresh=0.3)
    print("pred counts:", [pred["body_boxes_xyxy"].shape[0] for pred in preds])
