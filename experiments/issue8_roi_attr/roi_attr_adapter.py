from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from dog_yolov11 import DogYOLOv11, decode_dog_predictions
from experiments.issue8_roi_attr.roi_attr_head import DogRoiAttrHead


class DogYoloWithFeatures(nn.Module):
    """
    Non-invasive adapter: returns (preds, features) without changing DogYOLOv11 source.
    """

    def __init__(self, detector: DogYOLOv11):
        super().__init__()
        self.detector = detector

    def forward(
        self,
        images: torch.Tensor,
        *,
        return_features: bool = False,
    ) -> List[torch.Tensor] | Tuple[List[torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        p3, p4, p5 = self.detector.backbone(images)
        preds = [
            self.detector.head_s8(p3),
            self.detector.head_s16(p4),
            self.detector.head_s32(p5),
        ]
        if return_features:
            return preds, (p3, p4, p5)
        return preds


class RoiAttrExperimentModel(nn.Module):
    """
    Experimental model wrapper for issue #8:
    - Detection: existing DogYOLOv11
    - Attributes: ROIAlign-based DogRoiAttrHead
    """

    def __init__(self, detector: DogYOLOv11, roi_head: DogRoiAttrHead):
        super().__init__()
        self.detector = DogYoloWithFeatures(detector)
        self.roi_head = roi_head

    def forward(
        self,
        images: torch.Tensor,
        *,
        body_boxes: Optional[Sequence[torch.Tensor]] = None,
        head_boxes: Optional[Sequence[torch.Tensor]] = None,
        head_valid: Optional[Sequence[torch.Tensor]] = None,
        return_features: bool = False,
    ) -> Dict[str, Any]:
        preds, features = self.detector(images, return_features=True)
        out: Dict[str, Any] = {"preds": preds}
        if return_features:
            out["features"] = features

        if body_boxes is not None:
            out["roi"] = self.roi_head(
                features,
                body_boxes=body_boxes,
                head_boxes=head_boxes,
                head_valid=head_valid,
                image_size=(int(images.shape[-2]), int(images.shape[-1])),
            )

        return out

    @torch.no_grad()
    def infer_with_roi_attributes(
        self,
        images: torch.Tensor,
        *,
        breed_names: Sequence[str],
        emotion_names: Sequence[str],
        action_names: Sequence[str],
        obj_thres: float = 0.05,
        conf_thres: float = 0.25,
        iou_thres: float = 0.50,
        class_agnostic: bool = False,
        max_det: int = 300,
    ) -> List[List[Dict[str, Any]]]:
        preds, features = self.detector(images, return_features=True)
        decoded = decode_dog_predictions(
            preds,
            image_size=(int(images.shape[-2]), int(images.shape[-1])),
            breed_names=breed_names,
            emotion_names=emotion_names,
            action_names=action_names,
            obj_thres=obj_thres,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            apply_nms=True,
            class_agnostic=class_agnostic,
            max_det=max_det,
        )

        body_boxes: List[torch.Tensor] = []
        head_boxes: List[torch.Tensor] = []
        head_valid: List[torch.Tensor] = []
        device = images.device

        for records in decoded:
            if not records:
                body_boxes.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                head_boxes.append(torch.zeros((0, 4), dtype=torch.float32, device=device))
                head_valid.append(torch.zeros((0,), dtype=torch.bool, device=device))
                continue

            cur_body: List[List[float]] = []
            cur_head: List[List[float]] = []
            cur_head_valid: List[bool] = []
            for rec in records:
                b = rec.get("bodybndbox", [0, 0, 0, 0])
                h = rec.get("headbndbox", [0, 0, 0, 0])
                cur_body.append([float(b[0]), float(b[1]), float(b[2]), float(b[3])])
                cur_head.append([float(h[0]), float(h[1]), float(h[2]), float(h[3])])
                cur_head_valid.append(float(h[2]) > float(h[0]) and float(h[3]) > float(h[1]))

            body_boxes.append(torch.tensor(cur_body, dtype=torch.float32, device=device))
            head_boxes.append(torch.tensor(cur_head, dtype=torch.float32, device=device))
            head_valid.append(torch.tensor(cur_head_valid, dtype=torch.bool, device=device))

        roi_out = self.roi_head(
            features,
            body_boxes=body_boxes,
            head_boxes=head_boxes,
            head_valid=head_valid,
            image_size=(int(images.shape[-2]), int(images.shape[-1])),
        )
        if roi_out["emotion_logits"].shape[0] == 0:
            return decoded

        emo_idx = torch.argmax(roi_out["emotion_logits"], dim=1)
        act_idx = torch.argmax(roi_out["action_logits"], dim=1)
        breed_idx = torch.argmax(roi_out["breed_logits"], dim=1) if "breed_logits" in roi_out else None

        for ridx in range(roi_out["batch_indices"].shape[0]):
            b = int(roi_out["batch_indices"][ridx].item())
            o = int(roi_out["object_indices"][ridx].item())
            if b >= len(decoded) or o >= len(decoded[b]):
                continue
            decoded[b][o]["emotional"] = emotion_names[int(emo_idx[ridx].item())]
            decoded[b][o]["action"] = action_names[int(act_idx[ridx].item())]
            if breed_idx is not None:
                decoded[b][o]["label"] = breed_names[int(breed_idx[ridx].item())]

        return decoded
