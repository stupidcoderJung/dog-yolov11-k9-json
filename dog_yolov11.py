import math
from typing import Any, Dict, List, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Basic Blocks (YOLOv11 Nano Style)
# ============================================================


def autopad(k, p=None, d=1):
    if p is None:
        p = k // 2
    return p


def parse_image_size(img_size: Any) -> Tuple[float, float]:
    if isinstance(img_size, (int, float)):
        return float(img_size), float(img_size)
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        h, w = img_size
        return float(h), float(w)
    if torch.is_tensor(img_size):
        if img_size.numel() == 1:
            v = float(img_size.item())
            return v, v
        if img_size.numel() == 2:
            vals = img_size.detach().cpu().tolist()
            return float(vals[0]), float(vals[1])
    raise ValueError(f"Unsupported img_size format: {type(img_size)}")


def xyxy_to_coco(box: Sequence[float]) -> List[int]:
    x1, y1, x2, y2 = box
    return [
        int(round(x1)),
        int(round(y1)),
        int(round(max(0.0, x2 - x1))),
        int(round(max(0.0, y2 - y1))),
    ]


def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU for boxes in xyxy format.
    boxes1: [N,4], boxes2: [M,4]
    return: [N,M]
    """
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (boxes1.shape[0], boxes2.shape[0]), dtype=boxes1.dtype, device=boxes1.device
        )

    tl = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    br = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (br - tl).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def nms_keep_indices(
    boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float = 0.5
) -> torch.Tensor:
    """
    Pure torch NMS.
    boxes: [N,4] xyxy, scores: [N]
    return: kept indices over original N.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = torch.argsort(scores, descending=True)
    keep: List[torch.Tensor] = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = box_iou_xyxy(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        rest = rest[ious <= iou_thres]
        order = rest

    if len(keep) == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    return torch.stack(keep)


class Conv(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, g=1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = Conv(c, c // 2, 1)
        self.cv2 = Conv(c // 2, c, 3)

    def forward(self, x):
        return x + self.cv2(self.cv1(x))


class C3k2(nn.Module):
    def __init__(self, c, n=1):
        super().__init__()
        self.blocks = nn.Sequential(*[Bottleneck(c) for _ in range(n)])

    def forward(self, x):
        return self.blocks(x)


class SPPF(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.cv1 = Conv(c, c // 2, 1)
        self.pool = nn.MaxPool2d(5, 1, 2)
        self.cv2 = Conv(c * 2, c, 1)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        return self.cv2(torch.cat([x, y1, y2, self.pool(y2)], 1))


# ============================================================
# Backbone
# ============================================================


class YOLOv11NanoBackbone(nn.Module):
    def __init__(self, width_mult=0.23):
        super().__init__()
        base = int(64 * width_mult)

        self.stage1 = Conv(3, base, 3, 2)  # /2
        self.stage2 = Conv(base, base * 2, 3, 2)  # /4
        self.stage3 = nn.Sequential(
            Conv(base * 2, base * 4, 3, 2),  # /8
            C3k2(base * 4, 2),
        )
        self.stage4 = nn.Sequential(
            Conv(base * 4, base * 8, 3, 2),  # /16
            C3k2(base * 8, 2),
        )
        self.stage5 = nn.Sequential(
            Conv(base * 8, base * 12, 3, 2),  # /32
            C3k2(base * 12, 1),
            SPPF(base * 12),
        )

        self.out_channels = [base * 4, base * 8, base * 12]

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        p3 = self.stage3(x)  # stride 8
        p4 = self.stage4(p3)  # stride 16
        p5 = self.stage5(p4)  # stride 32
        return p3, p4, p5


# ============================================================
# Detection Head
# ============================================================


class DetectHead(nn.Module):
    def __init__(self, in_ch, num_breeds, num_emotions, num_actions):
        super().__init__()
        self.num_breeds = num_breeds
        self.num_emotions = num_emotions
        self.num_actions = num_actions

        self.out_dim = 1 + 4 + 4 + num_breeds + num_emotions + num_actions

        self.conv1 = Conv(in_ch, in_ch)
        self.conv2 = Conv(in_ch, in_ch)
        self.pred = nn.Conv2d(in_ch, self.out_dim, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pred(x)
        return x.permute(0, 2, 3, 1)


# ============================================================
# Full Model
# ============================================================


class DogYOLOv11(nn.Module):
    def __init__(
        self, num_breeds=120, num_emotions=5, num_actions=5, width_mult=0.23
    ):
        super().__init__()

        self.backbone = YOLOv11NanoBackbone(width_mult)

        chs = self.backbone.out_channels
        self.head_s8 = DetectHead(chs[0], num_breeds, num_emotions, num_actions)
        self.head_s16 = DetectHead(chs[1], num_breeds, num_emotions, num_actions)
        self.head_s32 = DetectHead(chs[2], num_breeds, num_emotions, num_actions)

        self.strides = [8, 16, 32]

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        o8 = self.head_s8(p3)
        o16 = self.head_s16(p4)
        o32 = self.head_s32(p5)
        return [o8, o16, o32]  # low→high (stride 8,16,32)


# ============================================================
# Loss (완전 정합 버전)
# ============================================================


class DogYOLOLoss(nn.Module):
    def __init__(
        self,
        num_breeds,
        num_emotions,
        num_actions,
        lambda_obj=1.0,
        lambda_box=5.0,
        lambda_head=2.0,
        lambda_attr=1.0,
        ignore_index=-100,
    ):
        super().__init__()

        self.num_breeds = num_breeds
        self.num_emotions = num_emotions
        self.num_actions = num_actions

        self.lambda_obj = lambda_obj
        self.lambda_box = lambda_box
        self.lambda_head = lambda_head
        self.lambda_attr = lambda_attr
        self.ignore_index = ignore_index

        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.smooth_l1 = nn.SmoothL1Loss(reduction="none")

    def _safe_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Unknown/ignored label is excluded from classification loss.
        if int(target.item()) == self.ignore_index:
            return torch.zeros((), device=logits.device)
        return F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

    def forward(self, preds, targets, img_size):
        img_h, img_w = parse_image_size(img_size)
        device = preds[0].device
        total_loss = torch.tensor(0.0, device=device)
        total_pos = 0

        for scale_idx, pred in enumerate(preds):
            B, H, W, _ = pred.shape
            stride_x = img_w / float(W)
            stride_y = img_h / float(H)

            obj = pred[..., 0]
            body_raw = pred[..., 1:5]
            head_raw = pred[..., 5:9]

            off = 9
            breed_logits = pred[..., off : off + self.num_breeds]
            off += self.num_breeds
            emo_logits = pred[..., off : off + self.num_emotions]
            off += self.num_emotions
            act_logits = pred[..., off : off + self.num_actions]

            obj_target = torch.zeros_like(obj)

            for b in range(B):
                if len(targets[b]["body_boxes"]) == 0:
                    continue

                body_boxes = targets[b]["body_boxes"].to(device)
                head_boxes = targets[b]["head_boxes"].to(device)
                labels = targets[b]["labels"].to(device)
                emotions = targets[b]["emotions"].to(device)
                actions = targets[b]["actions"].to(device)
                head_valid = targets[b].get("head_valid", None)
                if head_valid is not None:
                    head_valid = head_valid.to(device)
                else:
                    head_valid = torch.ones(
                        (len(body_boxes),), dtype=torch.bool, device=device
                    )

                # If multiple objects fall into the same grid cell, keep one target per cell.
                # Sort by smaller area first to avoid losing tiny dogs near larger ones.
                areas = (body_boxes[:, 2] - body_boxes[:, 0]).clamp(min=0) * (
                    body_boxes[:, 3] - body_boxes[:, 1]
                ).clamp(min=0)
                obj_indices = torch.argsort(areas).tolist()

                for i in obj_indices:
                    x1, y1, x2, y2 = body_boxes[i]
                    # Backward compatibility: if coords are normalized, convert to pixels.
                    if torch.max(body_boxes[i]) <= 1.5:
                        x1 = x1 * img_w
                        x2 = x2 * img_w
                        y1 = y1 * img_h
                        y2 = y2 * img_h

                    bw = x2 - x1
                    bh = y2 - y1
                    if bw <= 0 or bh <= 0:
                        continue

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    gx = int(cx / stride_x)
                    gy = int(cy / stride_y)

                    if gx < 0 or gy < 0 or gx >= W or gy >= H:
                        continue
                    if obj_target[b, gy, gx].item() > 0.5:
                        continue

                    obj_target[b, gy, gx] = 1.0
                    total_pos += 1

                    # body offset
                    tx = cx / stride_x - gx
                    ty = cy / stride_y - gy
                    tw = torch.log(bw / stride_x + 1e-6)
                    th = torch.log(bh / stride_y + 1e-6)

                    pred_xy = torch.sigmoid(body_raw[b, gy, gx, 0:2])
                    pred_wh = body_raw[b, gy, gx, 2:4]

                    total_loss += self.lambda_box * (
                        self.smooth_l1(
                            pred_xy, torch.tensor([tx, ty], device=device)
                        ).sum()
                        + self.smooth_l1(
                            pred_wh, torch.tensor([tw, th], device=device)
                        ).sum()
                    )

                    # head relative
                    if bool(head_valid[i].item()):
                        hx1, hy1, hx2, hy2 = head_boxes[i]
                        if torch.max(head_boxes[i]) <= 1.5:
                            hx1 = hx1 * img_w
                            hx2 = hx2 * img_w
                            hy1 = hy1 * img_h
                            hy2 = hy2 * img_h

                        hbw = hx2 - hx1
                        hbh = hy2 - hy1
                        if hbw > 0 and hbh > 0:
                            rel = torch.tensor(
                                [
                                    (hx1 - x1) / bw,
                                    (hy1 - y1) / bh,
                                    (hx2 - x1) / bw,
                                    (hy2 - y1) / bh,
                                ],
                                device=device,
                            ).clamp(0.0, 1.0)

                            pred_head = torch.sigmoid(head_raw[b, gy, gx])
                            total_loss += self.lambda_head * self.smooth_l1(
                                pred_head, rel
                            ).sum()

                    # attribute
                    total_loss += self.lambda_attr * (
                        self._safe_ce(breed_logits[b, gy, gx], labels[i])
                        + self._safe_ce(emo_logits[b, gy, gx], emotions[i])
                        + self._safe_ce(act_logits[b, gy, gx], actions[i])
                    )

            # objectness
            total_loss += self.lambda_obj * self.bce(obj, obj_target).sum()

        return total_loss / max(total_pos, 1)


def _map_label(
    value: str,
    mapping: Dict[str, int],
    unknown_policy: str = "ignore",
    ignore_index: int = -100,
    unknown_token: str = "Unknown",
) -> int:
    if value in mapping:
        return int(mapping[value])
    if unknown_policy == "class" and unknown_token in mapping:
        return int(mapping[unknown_token])
    return int(ignore_index)


def annotations_to_target(
    annotations: List[Dict[str, Any]],
    breed_to_idx: Dict[str, int],
    emotion_to_idx: Dict[str, int],
    action_to_idx: Dict[str, int],
    unknown_breed_policy: str = "ignore",
    ignore_index: int = -100,
) -> Dict[str, torch.Tensor]:
    """
    Convert one image's JSON annotations to DogYOLOLoss target format.
    Expected item keys include:
      label, bodybndbox[x1,y1,x2,y2], headbndbox[x1,y1,x2,y2], emotional, action
    """
    body_boxes: List[List[float]] = []
    head_boxes: List[List[float]] = []
    labels: List[int] = []
    emotions: List[int] = []
    actions: List[int] = []
    head_valid: List[bool] = []

    for ann in annotations:
        body = ann.get("bodybndbox", None)
        if body is None or len(body) != 4:
            continue

        x1, y1, x2, y2 = [float(v) for v in body]
        if x2 <= x1 or y2 <= y1:
            continue

        head = ann.get("headbndbox", [0, 0, 0, 0])
        if head is None or len(head) != 4:
            head = [0, 0, 0, 0]
        hx1, hy1, hx2, hy2 = [float(v) for v in head]
        is_head_valid = (hx2 > hx1) and (hy2 > hy1)

        body_boxes.append([x1, y1, x2, y2])
        if is_head_valid:
            head_boxes.append([hx1, hy1, hx2, hy2])
        else:
            head_boxes.append([0.0, 0.0, 0.0, 0.0])
        head_valid.append(is_head_valid)

        breed_name = str(ann.get("label", "Unknown"))
        emo_name = str(ann.get("emotional", "Unknown"))
        act_name = str(ann.get("action", "Unknown"))

        labels.append(
            _map_label(
                breed_name,
                breed_to_idx,
                unknown_policy=unknown_breed_policy,
                ignore_index=ignore_index,
            )
        )
        emotions.append(
            _map_label(
                emo_name,
                emotion_to_idx,
                unknown_policy="ignore",
                ignore_index=ignore_index,
            )
        )
        actions.append(
            _map_label(
                act_name,
                action_to_idx,
                unknown_policy="ignore",
                ignore_index=ignore_index,
            )
        )

    if len(body_boxes) == 0:
        return {
            "body_boxes": torch.zeros((0, 4), dtype=torch.float32),
            "head_boxes": torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.zeros((0,), dtype=torch.long),
            "emotions": torch.zeros((0,), dtype=torch.long),
            "actions": torch.zeros((0,), dtype=torch.long),
            "head_valid": torch.zeros((0,), dtype=torch.bool),
        }

    return {
        "body_boxes": torch.tensor(body_boxes, dtype=torch.float32),
        "head_boxes": torch.tensor(head_boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.long),
        "emotions": torch.tensor(emotions, dtype=torch.long),
        "actions": torch.tensor(actions, dtype=torch.long),
        "head_valid": torch.tensor(head_valid, dtype=torch.bool),
    }


@torch.no_grad()
def decode_dog_predictions(
    preds: List[torch.Tensor],
    image_size: Any,
    breed_names: Sequence[str],
    emotion_names: Sequence[str],
    action_names: Sequence[str],
    obj_thres: float = 0.05,
    conf_thres: float = 0.25,
    iou_thres: float = 0.50,
    apply_nms: bool = True,
    class_agnostic: bool = False,
    max_det: int = 300,
) -> List[List[Dict[str, Any]]]:
    """
    Decode model raw outputs to JSON-like records:
      label, bodybndbox, bodybndbox_coco, headbndbox, headbndbox_coco, emotional, action
    """
    img_h, img_w = parse_image_size(image_size)
    batch = preds[0].shape[0]
    results: List[List[Tuple[float, int, List[float], Dict[str, Any]]]] = [[] for _ in range(batch)]

    num_breeds = len(breed_names)
    num_emotions = len(emotion_names)
    num_actions = len(action_names)

    for pred in preds:
        B, H, W, _ = pred.shape
        stride_x = img_w / float(W)
        stride_y = img_h / float(H)

        obj = torch.sigmoid(pred[..., 0])
        body_xy = torch.sigmoid(pred[..., 1:3])
        body_wh = torch.exp(pred[..., 3:5])
        head_rel = torch.sigmoid(pred[..., 5:9])

        off = 9
        breed_logits = pred[..., off : off + num_breeds]
        off += num_breeds
        emo_logits = pred[..., off : off + num_emotions]
        off += num_emotions
        act_logits = pred[..., off : off + num_actions]

        for b in range(B):
            ys, xs = torch.where(obj[b] >= obj_thres)
            for gy, gx in zip(ys.tolist(), xs.tolist()):
                obj_score = float(obj[b, gy, gx].item())

                cx = (gx + float(body_xy[b, gy, gx, 0].item())) * stride_x
                cy = (gy + float(body_xy[b, gy, gx, 1].item())) * stride_y
                bw = float(body_wh[b, gy, gx, 0].item()) * stride_x
                bh = float(body_wh[b, gy, gx, 1].item()) * stride_y

                x1 = max(0.0, cx - bw / 2.0)
                y1 = max(0.0, cy - bh / 2.0)
                x2 = min(img_w - 1.0, cx + bw / 2.0)
                y2 = min(img_h - 1.0, cy + bh / 2.0)
                if x2 <= x1 or y2 <= y1:
                    continue

                rel = head_rel[b, gy, gx]
                hx1 = max(0.0, x1 + float(rel[0].item()) * (x2 - x1))
                hy1 = max(0.0, y1 + float(rel[1].item()) * (y2 - y1))
                hx2 = min(img_w - 1.0, x1 + float(rel[2].item()) * (x2 - x1))
                hy2 = min(img_h - 1.0, y1 + float(rel[3].item()) * (y2 - y1))
                if hx2 <= hx1 or hy2 <= hy1:
                    head_xyxy = [0.0, 0.0, 0.0, 0.0]
                else:
                    head_xyxy = [hx1, hy1, hx2, hy2]

                breed_probs = torch.softmax(breed_logits[b, gy, gx], dim=0)
                breed_conf, breed_idx_t = torch.max(breed_probs, dim=0)
                breed_idx = int(breed_idx_t.item())
                cls_score = float(breed_conf.item())
                score = obj_score * cls_score
                if score < conf_thres:
                    continue

                emo_idx = int(torch.argmax(emo_logits[b, gy, gx]).item())
                act_idx = int(torch.argmax(act_logits[b, gy, gx]).item())

                record = {
                    "label": breed_names[breed_idx]
                    if breed_idx < len(breed_names)
                    else f"class_{breed_idx}",
                    "bodybndbox": [
                        int(round(x1)),
                        int(round(y1)),
                        int(round(x2)),
                        int(round(y2)),
                    ],
                    "bodybndbox_coco": xyxy_to_coco([x1, y1, x2, y2]),
                    "headbndbox": [
                        int(round(head_xyxy[0])),
                        int(round(head_xyxy[1])),
                        int(round(head_xyxy[2])),
                        int(round(head_xyxy[3])),
                    ],
                    "headbndbox_coco": xyxy_to_coco(head_xyxy),
                    "emotional": emotion_names[emo_idx]
                    if emo_idx < len(emotion_names)
                    else f"emotion_{emo_idx}",
                    "action": action_names[act_idx]
                    if act_idx < len(action_names)
                    else f"action_{act_idx}",
                    "confidence": round(score, 6),
                    "objectness": round(obj_score, 6),
                    "breed_confidence": round(cls_score, 6),
                }
                results[b].append((score, breed_idx, [x1, y1, x2, y2], record))

    out: List[List[Dict[str, Any]]] = []
    for per_img in results:
        if len(per_img) == 0:
            out.append([])
            continue

        if apply_nms:
            boxes = torch.tensor([x[2] for x in per_img], dtype=torch.float32)
            scores = torch.tensor([x[0] for x in per_img], dtype=torch.float32)
            class_ids = torch.tensor([x[1] for x in per_img], dtype=torch.long)

            if class_agnostic:
                keep = nms_keep_indices(boxes, scores, iou_thres=iou_thres)
            else:
                keep_parts: List[torch.Tensor] = []
                for cls in class_ids.unique():
                    cls_idx = torch.where(class_ids == cls)[0]
                    cls_keep = nms_keep_indices(
                        boxes[cls_idx], scores[cls_idx], iou_thres=iou_thres
                    )
                    keep_parts.append(cls_idx[cls_keep])
                if len(keep_parts) == 0:
                    keep = torch.zeros((0,), dtype=torch.long)
                else:
                    keep = torch.cat(keep_parts, dim=0)

            per_img = [per_img[i] for i in keep.tolist()]

        per_img = sorted(per_img, key=lambda x: x[0], reverse=True)[:max_det]
        out.append([rec for _, _, _, rec in per_img])
    return out


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def model_size_report(
    num_breeds: int = 120,
    num_emotions: int = 5,
    num_actions: int = 5,
    width_mult: float = 0.23,
) -> Dict[str, Any]:
    model = DogYOLOv11(
        num_breeds=num_breeds,
        num_emotions=num_emotions,
        num_actions=num_actions,
        width_mult=width_mult,
    )
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    return {
        "width_mult": width_mult,
        "num_breeds": num_breeds,
        "num_emotions": num_emotions,
        "num_actions": num_actions,
        "total_params": total,
        "trainable_params": trainable,
    }
