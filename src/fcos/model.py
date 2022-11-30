# from https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


from functools import partial
from typing import List, Tuple, Union, Optional
from torch import nn as nn, topk

from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from torchvision.models.resnet import resnet50, resnet18, resnet34, resnet101, resnet152
from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
from torchvision.ops.feature_pyramid_network import LastLevelP6P7
from torchvision.ops import misc as misc_nn_ops
from torchvision.models.detection.backbone_utils import _validate_trainable_layers


from .fcos import FCOS
from .hierarchical_multilevel_fcos import FCOS as hierarchical_multilevel_FCOS

model_urls = {
    'maskrcnn_resnet50_fpn_coco':
    'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
}








def make_fcos_model(
    pretrained = True,
    num_classes:int = 2,
    # transform parameters
    min_size: int = 800,
    max_size: int = 1333,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    trainable_backbone_layers: Optional[int] = None,
    center_sampling_radius:float = 1.5,
    score_thresh:float = 0.2,
    detections_per_img: int = 100,
    topk_candidates: int = 1000,
    backbone = 'resnet50',
    **kwargs
):
    
    
    # load pretrained weights

    trainable_backbone_layers = _validate_trainable_layers(pretrained, trainable_backbone_layers, 5, 3)
    backbone = return_backbone(name=backbone, pretrained=pretrained)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )

    model = FCOS(
        backbone,
        num_classes,
        min_size = min_size,
        max_size = max_size,
        image_mean = image_mean,
        image_std = image_std,
        center_sampling_radius = center_sampling_radius,
        score_thresh = score_thresh,
        nms_thresh = 0.6,
        detections_per_img = detections_per_img,
        topk_candidates = topk_candidates,
        **kwargs
        )
    
    return model




def make_two_level_hierarchical_fcos_model(
    pretrained = True,
    num_classes:int = 2,
    num_sub_classes:int = 8,
    num_sub_classes_level2:int=4,
    # transform parameters
    min_size: int = 800,
    max_size: int = 1333,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
    trainable_backbone_layers: Optional[int] = None,
    center_sampling_radius:float = 1.5,
    score_thresh:float = 0.2,
    detections_per_img: int = 100,
    topk_candidates: int = 1000,
    backbone = 'resnet50',
    **kwargs
):
    
    
    # load pretrained weights

    trainable_backbone_layers = _validate_trainable_layers(pretrained, trainable_backbone_layers, 5, 3)
    backbone = return_backbone(name=backbone, pretrained=pretrained)
    backbone = _resnet_fpn_extractor(
        backbone, trainable_backbone_layers, returned_layers=[2, 3, 4], extra_blocks=LastLevelP6P7(256, 256)
    )

    model = hierarchical_multilevel_FCOS(
        backbone,
        num_classes,
        num_sub_classes,
        min_size = min_size,
        num_sub_classes_level2=num_sub_classes_level2,
        max_size = max_size,
        image_mean = image_mean,
        image_std = image_std,
        center_sampling_radius = center_sampling_radius,
        score_thresh = score_thresh,
        nms_thresh = 0.6,
        detections_per_img = detections_per_img,
        topk_candidates = topk_candidates,
        **kwargs
        )
    
    return model



def return_backbone(name:str = 'resnet50', pretrained: bool = True) -> resnet_fpn_backbone:
    
    supported_backbones = [
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101',
        'resnet152'
    ]
    
    # check for wright name
    assert name in supported_backbones, f"{name} not supported, choose one of these: {supported_backbones}"

    if name == 'resnet18':
        backbone = resnet18(pretrained,progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    elif name == 'resnet34':
        backbone = resnet34(pretrained,progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    elif name == 'resnet50':
        backbone = resnet50(pretrained,progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    elif name == 'resnet101':
        backbone = resnet101(pretrained,progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    elif name == 'resnet152':
        backbone = resnet152(pretrained,progress=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)

    return backbone