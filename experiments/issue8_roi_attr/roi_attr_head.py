from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from torchvision.ops import roi_align
except Exception:  # pragma: no cover
    roi_align = None


def _as_box_tensor(boxes: torch.Tensor | Sequence[Sequence[float]], device: torch.device) -> torch.Tensor:
    if isinstance(boxes, torch.Tensor):
        out = boxes.to(device=device, dtype=torch.float32)
    else:
        out = torch.tensor(boxes, dtype=torch.float32, device=device)
    if out.numel() == 0:
        return out.reshape(0, 4)
    if out.ndim != 2 or out.shape[1] != 4:
        raise ValueError(f"Expected Nx4 boxes, got shape={tuple(out.shape)}")
    return out


def _clip_xyxy(boxes: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
    h, w = image_size
    clipped = boxes.clone()
    clipped[:, 0] = clipped[:, 0].clamp(0, max(w - 1, 0))
    clipped[:, 1] = clipped[:, 1].clamp(0, max(h - 1, 0))
    clipped[:, 2] = clipped[:, 2].clamp(0, max(w - 1, 0))
    clipped[:, 3] = clipped[:, 3].clamp(0, max(h - 1, 0))
    return clipped


class DogRoiAttrHead(nn.Module):
    """
    ROIAlign-based object-centric attribute head.

    Inputs are image-space body/head boxes and backbone feature maps.
    The head supports:
    - single-scale ROI (p3 only)
    - optional multi-scale ROI assignment (p3/p4/p5)
    - fusion mode: concat MLP or tiny cross-attention
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        num_emotions: int,
        num_actions: int,
        num_breeds: Optional[int] = None,
        *,
        roi_output_size: int = 7,
        hidden_dim: int = 192,
        fusion: str = "concat",
        use_multiscale: bool = False,
        feature_strides: Sequence[int] = (8, 16, 32),
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if len(in_channels) == 0:
            raise ValueError("in_channels must be non-empty")
        if len(in_channels) != len(feature_strides):
            raise ValueError("in_channels and feature_strides length mismatch")
        if fusion not in {"concat", "xattn"}:
            raise ValueError("fusion must be one of: concat, xattn")
        if fusion == "xattn" and hidden_dim % attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads for xattn")

        self.num_emotions = int(num_emotions)
        self.num_actions = int(num_actions)
        self.num_breeds = int(num_breeds) if num_breeds is not None else None

        self.roi_output_size = int(roi_output_size)
        self.hidden_dim = int(hidden_dim)
        self.fusion = fusion
        self.use_multiscale = bool(use_multiscale)
        self.feature_strides = tuple(int(s) for s in feature_strides)

        self.proj = nn.ModuleList(
            [nn.Conv2d(int(c), self.hidden_dim, kernel_size=1) for c in in_channels]
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        if self.fusion == "concat":
            self.concat_fuse = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(self.hidden_dim),
            )
        else:
            self.xattn_norm = nn.LayerNorm(self.hidden_dim)
            self.xattn = nn.MultiheadAttention(
                embed_dim=self.hidden_dim,
                num_heads=attention_heads,
                batch_first=True,
            )
            self.xffn = nn.Sequential(
                nn.LayerNorm(self.hidden_dim),
                nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                nn.GELU(),
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            )

        self.emotion_head = nn.Linear(self.hidden_dim, self.num_emotions)
        self.action_head = nn.Linear(self.hidden_dim, self.num_actions)
        if self.num_breeds is not None:
            self.breed_head = nn.Linear(self.hidden_dim, self.num_breeds)

    def _assign_levels(self, boxes_xyxy: torch.Tensor) -> torch.Tensor:
        """Assign ROI levels from ROI scale to nearest configured feature stride."""
        if boxes_xyxy.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=boxes_xyxy.device)

        w = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=1.0)
        h = (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=1.0)
        scale = torch.sqrt(w * h)

        # Map object scale to stride domain and choose nearest stride level.
        strides = torch.tensor(self.feature_strides, dtype=scale.dtype, device=scale.device)
        target_stride = (scale / 8.0).clamp(min=float(strides.min().item()), max=float(strides.max().item()))
        dist = torch.abs(torch.log2(target_stride[:, None]) - torch.log2(strides[None, :]))
        return torch.argmin(dist, dim=1)

    def _flatten_rois(
        self,
        boxes_per_image: Sequence[torch.Tensor | Sequence[Sequence[float]]],
        image_size: Tuple[int, int],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows: List[torch.Tensor] = []
        batch_indices: List[int] = []
        object_indices: List[int] = []

        for bidx, boxes in enumerate(boxes_per_image):
            cur = _as_box_tensor(boxes, device=device)
            if cur.numel() == 0:
                continue
            cur = _clip_xyxy(cur, image_size)
            valid = (cur[:, 2] > cur[:, 0]) & (cur[:, 3] > cur[:, 1])
            kept = torch.where(valid)[0]
            if kept.numel() == 0:
                continue

            batch_col = torch.full((kept.numel(), 1), float(bidx), device=device)
            rows.append(torch.cat([batch_col, cur[kept]], dim=1))
            batch_indices.extend([bidx] * int(kept.numel()))
            object_indices.extend([int(i) for i in kept.tolist()])

        if not rows:
            return (
                torch.zeros((0, 5), dtype=torch.float32, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
                torch.zeros((0,), dtype=torch.long, device=device),
            )

        rois = torch.cat(rows, dim=0)
        return (
            rois,
            torch.tensor(batch_indices, dtype=torch.long, device=device),
            torch.tensor(object_indices, dtype=torch.long, device=device),
        )

    def _extract_roi_features(self, projected: List[torch.Tensor], rois: torch.Tensor) -> torch.Tensor:
        if roi_align is None:
            raise RuntimeError(
                "torchvision.ops.roi_align is not available. Install torchvision with ops support."
            )
        if rois.numel() == 0:
            c = projected[0].shape[1]
            return torch.zeros((0, c, self.roi_output_size, self.roi_output_size), device=projected[0].device)

        if not self.use_multiscale:
            return roi_align(
                projected[0],
                rois,
                output_size=self.roi_output_size,
                spatial_scale=1.0 / float(self.feature_strides[0]),
                aligned=True,
            )

        levels = self._assign_levels(rois[:, 1:5])
        k = rois.shape[0]
        c = projected[0].shape[1]
        out = torch.zeros(
            (k, c, self.roi_output_size, self.roi_output_size),
            dtype=projected[0].dtype,
            device=projected[0].device,
        )

        for lvl in range(len(projected)):
            mask = levels == lvl
            if not torch.any(mask):
                continue
            roi_lvl = rois[mask]
            feat_lvl = roi_align(
                projected[lvl],
                roi_lvl,
                output_size=self.roi_output_size,
                spatial_scale=1.0 / float(self.feature_strides[lvl]),
                aligned=True,
            )
            out[mask] = feat_lvl
        return out

    def forward(
        self,
        features: Sequence[torch.Tensor],
        body_boxes: Sequence[torch.Tensor | Sequence[Sequence[float]]],
        head_boxes: Optional[Sequence[torch.Tensor | Sequence[Sequence[float]]]] = None,
        head_valid: Optional[Sequence[torch.Tensor | Sequence[bool]]] = None,
        *,
        image_size: Tuple[int, int],
    ) -> Dict[str, torch.Tensor]:
        if len(features) != len(self.proj):
            raise ValueError(
                f"feature count mismatch: expected {len(self.proj)}, got {len(features)}"
            )
        device = features[0].device
        projected = [proj(feat) for proj, feat in zip(self.proj, features)]

        body_rois, batch_idx, obj_idx = self._flatten_rois(body_boxes, image_size, device=device)
        if body_rois.numel() == 0:
            out: Dict[str, torch.Tensor] = {
                "emotion_logits": torch.zeros((0, self.num_emotions), device=device),
                "action_logits": torch.zeros((0, self.num_actions), device=device),
                "batch_indices": batch_idx,
                "object_indices": obj_idx,
            }
            if self.num_breeds is not None:
                out["breed_logits"] = torch.zeros((0, self.num_breeds), device=device)
            return out

        body_feat = self._extract_roi_features(projected, body_rois)
        body_vec = self.pool(body_feat).flatten(1)

        # Build per-object head rois aligned to (batch_idx, obj_idx).
        head_vec = torch.zeros_like(body_vec)
        if head_boxes is not None:
            head_rows: List[List[float]] = []
            valid_rows: List[int] = []
            for ridx in range(body_rois.shape[0]):
                b = int(batch_idx[ridx].item())
                o = int(obj_idx[ridx].item())
                hboxes = _as_box_tensor(head_boxes[b], device=device)
                if o >= hboxes.shape[0]:
                    continue

                use_head = True
                if head_valid is not None:
                    hv = head_valid[b]
                    hv_tensor = hv.to(device=device) if isinstance(hv, torch.Tensor) else torch.tensor(hv, device=device)
                    if o >= hv_tensor.shape[0] or not bool(hv_tensor[o].item()):
                        use_head = False

                head = hboxes[o : o + 1]
                head = _clip_xyxy(head, image_size)
                if not (head[0, 2] > head[0, 0] and head[0, 3] > head[0, 1]):
                    use_head = False

                if not use_head:
                    continue

                head_rows.append([float(b), float(head[0, 0]), float(head[0, 1]), float(head[0, 2]), float(head[0, 3])])
                valid_rows.append(ridx)

            if head_rows:
                head_rois = torch.tensor(head_rows, dtype=torch.float32, device=device)
                head_feat = self._extract_roi_features(projected, head_rois)
                head_tokens = self.pool(head_feat).flatten(1)
                dst = torch.tensor(valid_rows, dtype=torch.long, device=device)
                head_vec[dst] = head_tokens

        if self.fusion == "concat":
            fused = self.concat_fuse(torch.cat([body_vec, head_vec], dim=1))
        else:
            tokens = torch.stack([body_vec, head_vec], dim=1)
            normed = self.xattn_norm(tokens)
            attn_out, _ = self.xattn(normed, normed, normed, need_weights=False)
            tokens = tokens + attn_out
            tokens = tokens + self.xffn(tokens)
            fused = tokens.mean(dim=1)

        out = {
            "emotion_logits": self.emotion_head(fused),
            "action_logits": self.action_head(fused),
            "batch_indices": batch_idx,
            "object_indices": obj_idx,
        }
        if self.num_breeds is not None:
            out["breed_logits"] = self.breed_head(fused)
        return out
