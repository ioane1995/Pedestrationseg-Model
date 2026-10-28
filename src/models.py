"""Model factory.

* maskrcnn_r50       - Mask R-CNN, ResNet-50-FPN, COCO-pretrained (accuracy-oriented, ~44M params)
* maskrcnn_mobilenet - Mask R-CNN on a MobileNetV3-Large-FPN backbone. Backbone/RPN/box-head come
                       from the COCO-pretrained Faster R-CNN-MobileNetV3; the mask branch is new
                       (speed/size-oriented alternative).
"""
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn, maskrcnn_resnet50_fpn)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNN, MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

MODEL_NAMES = ["maskrcnn_r50", "maskrcnn_mobilenet"]


def build_model(name="maskrcnn_r50", num_classes=2, pretrained=True, trainable_layers=3,
                min_size=None, max_size=None, hidden=256):
    """num_classes includes background (pedestrian task -> 2).
    trainable_layers: how many backbone stages are fine-tuned (0 = frozen backbone, 5 = all)."""
    size_kw = {}
    if min_size:
        size_kw["min_size"] = int(min_size)
    if max_size:
        size_kw["max_size"] = int(max_size)

    if name == "maskrcnn_r50":
        if pretrained:
            kw = dict(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT, trainable_backbone_layers=trainable_layers)
        else:
            kw = dict(weights=None, weights_backbone=None)
        model = maskrcnn_resnet50_fpn(**kw, **size_kw)
        in_f = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_f, num_classes)
        in_c = model.roi_heads.mask_predictor.conv5_mask.in_channels
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_c, hidden, num_classes)
        return model

    if name == "maskrcnn_mobilenet":
        if pretrained:
            kw = dict(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT,
                      trainable_backbone_layers=trainable_layers)
        else:
            kw = dict(weights=None, weights_backbone=None)
        det = fasterrcnn_mobilenet_v3_large_fpn(**kw)
        sizes = {"min_size": 800, "max_size": 1333}
        sizes.update(size_kw)
        model = MaskRCNN(
            det.backbone, num_classes=num_classes,
            rpn_anchor_generator=det.rpn.anchor_generator,
            box_roi_pool=MultiScaleRoIAlign(["0", "1"], 7, 2),
            mask_roi_pool=MultiScaleRoIAlign(["0", "1"], 14, 2),
            **sizes)
        model.rpn = det.rpn                                # reuse pretrained RPN
        model.roi_heads.box_head = det.roi_heads.box_head  # reuse pretrained box head
        return model

    raise ValueError(f"unknown model {name}; choose from {MODEL_NAMES}")
