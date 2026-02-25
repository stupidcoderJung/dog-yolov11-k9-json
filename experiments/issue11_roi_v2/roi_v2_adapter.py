from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn

from dog_yolov11 import decode_dog_predictions, nms_keep_indices
from experiments.issue8_roi_attr.roi_attr_adapter import DogYoloWithFeatures
from experiments.issue8_roi_attr.roi_attr_head import DogRoiAttrHead
from experiments.issue11_roi_v2.calibration import apply_temperature_to_probability


class RoiV2HybridExperimentModel(nn.Module):
    """
    Issue #11 hybrid experiment wrapper.

    Key additions over issue8 wrapper:
    - score_policy to compare confidence strategies
    - explicit score component logging (objectness/breed/final)
    - optional temperature scaling for final confidence calibration
    """

    def __init__(
        self,
        detector: nn.Module,
        roi_head: DogRoiAttrHead,
        *,
        score_policy: str = "obj_x_breed",
        calibrated_temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if score_policy not in {"obj_x_breed", "calibrated_obj_x_breed", "breed_only"}:
            raise ValueError(
                "score_policy must be one of: obj_x_breed, calibrated_obj_x_breed, breed_only"
            )
        self.detector = DogYoloWithFeatures(detector)
        self.roi_head = roi_head
        self.score_policy = score_policy
        self.calibrated_temperature = max(float(calibrated_temperature), 1e-3)

    def set_score_policy(self, score_policy: str) -> None:
        if score_policy not in {"obj_x_breed", "calibrated_obj_x_breed", "breed_only"}:
            raise ValueError(
                "score_policy must be one of: obj_x_breed, calibrated_obj_x_breed, breed_only"
            )
        self.score_policy = score_policy

    def set_calibrated_temperature(self, temperature: float) -> None:
        self.calibrated_temperature = max(float(temperature), 1e-3)

    def forward(
        self,
        images: torch.Tensor,
        *,
        body_boxes: Sequence[torch.Tensor] | None = None,
        head_boxes: Sequence[torch.Tensor] | None = None,
        head_valid: Sequence[torch.Tensor] | None = None,
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

    def _compose_final_confidence(self, objectness: float, breed_confidence: float) -> float:
        obj = max(0.0, min(1.0, float(objectness)))
        breed = max(0.0, min(1.0, float(breed_confidence)))
        raw = obj * breed

        if self.score_policy == "breed_only":
            return breed
        if self.score_policy == "calibrated_obj_x_breed":
            calibrated = apply_temperature_to_probability(
                torch.tensor([raw], dtype=torch.float32),
                self.calibrated_temperature,
            )[0]
            return float(calibrated.item())
        return raw

    def _post_relabel_nms(
        self,
        decoded: List[List[Dict[str, Any]]],
        *,
        conf_thres: float,
        iou_thres: float,
        max_det: int,
        class_agnostic: bool,
    ) -> List[List[Dict[str, Any]]]:
        out: List[List[Dict[str, Any]]] = []
        for per_img in decoded:
            filtered = [
                rec for rec in per_img if float(rec.get("confidence", 0.0)) >= float(conf_thres)
            ]
            if not filtered:
                out.append([])
                continue

            boxes = torch.tensor(
                [rec.get("_body_xyxy", rec.get("bodybndbox", [0, 0, 0, 0])) for rec in filtered],
                dtype=torch.float32,
            )
            scores = torch.tensor(
                [float(rec.get("confidence", 0.0)) for rec in filtered],
                dtype=torch.float32,
            )

            if class_agnostic:
                keep = nms_keep_indices(boxes, scores, iou_thres=iou_thres)
            else:
                label_to_id: Dict[str, int] = {}
                class_ids: List[int] = []
                for rec in filtered:
                    label = str(rec.get("label", ""))
                    if label not in label_to_id:
                        label_to_id[label] = len(label_to_id)
                    class_ids.append(label_to_id[label])
                class_ids_t = torch.tensor(class_ids, dtype=torch.long)

                keep_parts: List[torch.Tensor] = []
                for cls in class_ids_t.unique():
                    cls_idx = torch.where(class_ids_t == cls)[0]
                    cls_keep = nms_keep_indices(boxes[cls_idx], scores[cls_idx], iou_thres=iou_thres)
                    keep_parts.append(cls_idx[cls_keep])
                keep = (
                    torch.cat(keep_parts, dim=0)
                    if keep_parts
                    else torch.zeros((0,), dtype=torch.long)
                )

            kept_records = [filtered[i] for i in keep.tolist()]
            kept_records = sorted(
                kept_records,
                key=lambda rec: float(rec.get("confidence", 0.0)),
                reverse=True,
            )[:max_det]
            out.append(kept_records)
        return out

    def _strip_internal_fields(self, decoded: List[List[Dict[str, Any]]]) -> List[List[Dict[str, Any]]]:
        for per_img in decoded:
            for rec in per_img:
                rec.pop("_body_xyxy", None)
        return decoded

    def _attach_score_components(self, decoded: List[List[Dict[str, Any]]]) -> None:
        for per_img in decoded:
            for rec in per_img:
                obj = float(rec.get("objectness", 1.0))
                breed = float(rec.get("breed_confidence", rec.get("confidence", 0.0)))
                final = self._compose_final_confidence(obj, breed)
                rec["objectness"] = round(obj, 6)
                rec["breed_confidence"] = round(breed, 6)
                rec["final_confidence"] = round(final, 6)
                rec["confidence"] = round(final, 6)
                rec["score_policy"] = self.score_policy
                if self.score_policy == "calibrated_obj_x_breed":
                    rec["calibrated_temperature"] = round(self.calibrated_temperature, 6)

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

        defer_nms = self.roi_head.num_breeds is not None
        decode_conf_thres = 0.0 if defer_nms else conf_thres
        decode_max_det = max_det
        if defer_nms:
            decode_max_det = int(sum(int(p.shape[1]) * int(p.shape[2]) for p in preds))

        decoded = decode_dog_predictions(
            preds,
            image_size=(int(images.shape[-2]), int(images.shape[-1])),
            breed_names=breed_names,
            emotion_names=emotion_names,
            action_names=action_names,
            obj_thres=obj_thres,
            conf_thres=decode_conf_thres,
            iou_thres=iou_thres,
            apply_nms=not defer_nms,
            class_agnostic=class_agnostic,
            max_det=decode_max_det,
            include_raw_boxes=defer_nms,
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

        if roi_out["emotion_logits"].shape[0] > 0:
            emo_idx = torch.argmax(roi_out["emotion_logits"], dim=1)
            act_idx = torch.argmax(roi_out["action_logits"], dim=1)
            breed_idx = None
            breed_conf = None
            if "breed_logits" in roi_out:
                breed_probs = torch.softmax(roi_out["breed_logits"], dim=1)
                breed_conf, breed_idx = torch.max(breed_probs, dim=1)

            for ridx in range(roi_out["batch_indices"].shape[0]):
                b = int(roi_out["batch_indices"][ridx].item())
                o = int(roi_out["object_indices"][ridx].item())
                if b >= len(decoded) or o >= len(decoded[b]):
                    continue
                emo_i = int(emo_idx[ridx].item())
                act_i = int(act_idx[ridx].item())
                decoded[b][o]["emotional"] = (
                    emotion_names[emo_i] if 0 <= emo_i < len(emotion_names) else f"emotion_{emo_i}"
                )
                decoded[b][o]["action"] = (
                    action_names[act_i] if 0 <= act_i < len(action_names) else f"action_{act_i}"
                )
                if breed_idx is not None and breed_conf is not None:
                    breed_i = int(breed_idx[ridx].item())
                    decoded[b][o]["label"] = (
                        breed_names[breed_i] if 0 <= breed_i < len(breed_names) else f"class_{breed_i}"
                    )
                    decoded[b][o]["breed_confidence"] = float(breed_conf[ridx].item())

        self._attach_score_components(decoded)
        if defer_nms:
            decoded = self._post_relabel_nms(
                decoded,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                max_det=max_det,
                class_agnostic=class_agnostic,
            )
        return self._strip_internal_fields(decoded)
