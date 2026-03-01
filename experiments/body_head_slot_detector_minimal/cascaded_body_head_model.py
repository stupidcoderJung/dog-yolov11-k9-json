#!/usr/bin/env python3
"""
Cascaded Body-Head Slot Detector

Key improvements:
1. Body detection first
2. ROI Pooling on body regions
3. Head detection within body context
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops import roi_align


class CascadedBodyHeadDetector(nn.Module):
    """
    Cascaded 2-stage detector:
    - Stage 1: Detect body boxes
    - Stage 2: Detect head boxes within body ROIs
    """

    def __init__(self, num_queries: int = 10, backbone_type: str = 'resnet18', use_pretrained: bool = False):
        super().__init__()
        self.num_queries = num_queries

        # Backbone: ResNet18 (pretrained)
        if backbone_type == 'resnet18':
            weights = ResNet18_Weights.DEFAULT if use_pretrained else None
            resnet = resnet18(weights=weights)
            # Remove FC layer
            self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # Up to avgpool
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")

        # Stage 1: Body detector
        self.body_objectness = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_queries)
        )

        self.body_regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_queries * 4)
        )

        # Stage 2: Head detector (with body context)
        # ROI feature는 7x7 → flatten → 512*7*7
        roi_feat_dim = self.feature_dim * 7 * 7

        self.head_regressor = nn.Sequential(
            nn.Linear(roi_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4)  # per ROI, 1 head box
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) image tensor

        Returns:
            dict:
                pred_logits: (B, Q) objectness logits
                pred_body_boxes: (B, Q, 4) body boxes (normalized cxcywh)
                pred_head_boxes: (B, Q, 4) head boxes (normalized cxcywh)
        """
        batch_size = x.shape[0]

        # Backbone feature extraction
        feat = self.backbone(x)  # (B, 512, H', W')

        # Global average pooling for body detection
        pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)  # (B, 512)

        # Stage 1: Body detection
        body_logits = self.body_objectness(pooled)  # (B, Q)
        body_boxes = self.body_regressor(pooled)  # (B, Q*4)
        body_boxes = body_boxes.view(batch_size, self.num_queries, 4).sigmoid()

        # Stage 2: Head detection within body ROIs
        # ROI Align: extract 7x7 features for each body box
        # Convert normalized cxcywh to xyxy for roi_align
        # roi_align expects (x1, y1, x2, y2) in original image coordinates
        h_feat, w_feat = feat.shape[2], feat.shape[3]

        head_boxes_list = []
        for b_idx in range(batch_size):
            # Body boxes for this image (Q, 4)
            body_cxcywh = body_boxes[b_idx]  # (Q, 4) normalized

            # Convert to xyxy (unnormalized, feature map scale)
            cx, cy, w, h = body_cxcywh.unbind(-1)
            x1 = (cx - w / 2) * w_feat
            y1 = (cy - h / 2) * h_feat
            x2 = (cx + w / 2) * w_feat
            y2 = (cy + h / 2) * h_feat
            body_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # (Q, 4)

            # Add batch index for roi_align.
            # We pass a single-image feature map (shape: 1, C, H, W), so ROI batch idx must be 0.
            batch_indices = torch.zeros((self.num_queries, 1), dtype=torch.float32, device=x.device)
            rois = torch.cat([batch_indices, body_xyxy], dim=1)  # (Q, 5)

            # ROI Align
            roi_feats = roi_align(
                feat[b_idx:b_idx+1],  # (1, C, H, W)
                rois,
                output_size=7,
                spatial_scale=1.0,
                aligned=True
            )  # (Q, 512, 7, 7)

            # Flatten
            roi_feats_flat = roi_feats.flatten(1)  # (Q, 512*7*7)

            # Predict head boxes
            head_boxes_b = self.head_regressor(roi_feats_flat)  # (Q, 4)
            head_boxes_b = head_boxes_b.sigmoid()  # Normalize

            head_boxes_list.append(head_boxes_b)

        head_boxes = torch.stack(head_boxes_list, dim=0)  # (B, Q, 4)

        return {
            'pred_logits': body_logits,
            'pred_body_boxes': body_boxes,
            'pred_head_boxes': head_boxes,
        }


def cascaded_body_head_loss(
    outputs,
    targets,
    w_obj: float = 1.0,
    w_body: float = 4.0,
    w_head: float = 20.0,  # Head에 더 큰 weight
    w_iou: float = 2.0,
    obj_pos_weight: float = 30.0,
    **kwargs
):
    """
    Same loss as body_head_set_loss but for cascaded model.
    Reuse the existing loss function from body_head_slot_model.
    """
    try:
        from .body_head_slot_model import body_head_set_loss
    except ImportError:
        from body_head_slot_model import body_head_set_loss
    return body_head_set_loss(
        outputs,
        targets,
        w_obj=w_obj,
        w_body=w_body,
        w_head=w_head,
        w_iou=w_iou,
        obj_pos_weight=obj_pos_weight,
        **kwargs
    )


if __name__ == '__main__':
    # Test
    model = CascadedBodyHeadDetector(num_queries=10)
    x = torch.randn(2, 3, 256, 256)
    outputs = model(x)
    print('Outputs:', {k: v.shape for k, v in outputs.items()})

    # Dummy targets
    targets = [
        {
            'body_boxes': torch.tensor([[0.5, 0.5, 0.4, 0.6]]),
            'head_boxes': torch.tensor([[0.5, 0.3, 0.15, 0.15]]),
        },
        {
            'body_boxes': torch.tensor([[0.6, 0.6, 0.3, 0.5]]),
            'head_boxes': torch.tensor([[0.6, 0.4, 0.12, 0.12]]),
        }
    ]

    loss_dict = cascaded_body_head_loss(outputs, targets, return_details=True)
    print('Loss:', {k: v.item() if hasattr(v, 'item') else v for k, v in loss_dict.items()})
