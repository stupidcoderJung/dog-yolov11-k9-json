from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoiAttributeLoss(nn.Module):
    """
    Loss for ROI attribute logits.

    Expected roi_outputs keys:
    - emotion_logits: [K, num_emotions]
    - action_logits: [K, num_actions]
    - optional breed_logits: [K, num_breeds]
    - batch_indices: [K]
    - object_indices: [K]

    Expected targets format:
    list[dict] where each dict has tensors:
    - labels
    - emotions
    - actions
    """

    def __init__(
        self,
        *,
        lambda_attr_roi: float = 1.0,
        ignore_index: int = -100,
        with_breed_head: bool = False,
    ) -> None:
        super().__init__()
        self.lambda_attr_roi = float(lambda_attr_roi)
        self.ignore_index = int(ignore_index)
        self.with_breed_head = bool(with_breed_head)

    def _gather(
        self,
        targets: Sequence[Dict[str, torch.Tensor]],
        batch_indices: torch.Tensor,
        object_indices: torch.Tensor,
        key: str,
    ) -> torch.Tensor:
        out: List[int] = []
        for ridx in range(batch_indices.numel()):
            b = int(batch_indices[ridx].item())
            o = int(object_indices[ridx].item())
            if b >= len(targets):
                out.append(self.ignore_index)
                continue
            cur = targets[b].get(key)
            if cur is None or o >= cur.shape[0]:
                out.append(self.ignore_index)
                continue
            out.append(int(cur[o].item()))
        return torch.tensor(out, dtype=torch.long, device=batch_indices.device)

    def _masked_ce(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.new_zeros(())
        valid = target != self.ignore_index
        if not torch.any(valid):
            return logits.new_zeros(())
        return F.cross_entropy(logits[valid], target[valid])

    def forward(
        self,
        roi_outputs: Dict[str, torch.Tensor],
        targets: Sequence[Dict[str, torch.Tensor]],
    ) -> Dict[str, torch.Tensor]:
        batch_idx = roi_outputs["batch_indices"]
        obj_idx = roi_outputs["object_indices"]

        target_emotions = self._gather(targets, batch_idx, obj_idx, "emotions")
        target_actions = self._gather(targets, batch_idx, obj_idx, "actions")

        loss_emotion = self._masked_ce(roi_outputs["emotion_logits"], target_emotions)
        loss_action = self._masked_ce(roi_outputs["action_logits"], target_actions)

        total = loss_emotion + loss_action
        loss_breed = total.new_zeros(())

        if self.with_breed_head:
            if "breed_logits" not in roi_outputs:
                raise ValueError(
                    "RoiAttributeLoss(with_breed_head=True) requires roi_outputs['breed_logits']"
                )
            target_breed = self._gather(targets, batch_idx, obj_idx, "labels")
            loss_breed = self._masked_ce(roi_outputs["breed_logits"], target_breed)
            total = total + loss_breed

        total = total * self.lambda_attr_roi
        return {
            "loss": total,
            "loss_emotion": loss_emotion,
            "loss_action": loss_action,
            "loss_breed": loss_breed,
        }


def combine_grid_and_roi_attr_loss(
    *,
    grid_attr_loss: torch.Tensor,
    roi_attr_loss: torch.Tensor,
    lambda_attr_grid: float,
    lambda_attr_roi: float,
) -> torch.Tensor:
    return float(lambda_attr_grid) * grid_attr_loss + float(lambda_attr_roi) * roi_attr_loss
