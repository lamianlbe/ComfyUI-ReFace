"""SCHP (Self-Correction Human Parsing) for head segmentation.

Uses the SCHP model with LIP dataset for full-body head parsing.
Model: exp-schp-201908261155-lip.pth
From: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

LIP label map (20 classes):
  0: Background, 1: Hat, 2: Hair, 3: Glove, 4: Sunglasses,
  5: Upper-clothes, 6: Dress, 7: Coat, 8: Socks, 9: Pants,
  10: Jumpsuits, 11: Scarf, 12: Skirt, 13: Face, 14: Left-arm,
  15: Right-arm, 16: Left-leg, 17: Right-leg, 18: Left-shoe, 19: Right-shoe
"""

import os
import functools
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict

import folder_paths

# Head labels in LIP: Hat(1) + Hair(2) + Sunglasses(4) + Face(13)
LIP_HEAD_LABELS = {1, 2, 4, 13}
LIP_INPUT_SIZE = [473, 473]
LIP_NUM_CLASSES = 20

# Head label in Pascal-Person-Part: Head(1)
PASCAL_HEAD_LABELS = {1}
PASCAL_INPUT_SIZE = [512, 512]
PASCAL_NUM_CLASSES = 7

_lip_model = None
_pascal_model = None
_device = None

SCHP_MODEL_DIR = "reface"
LIP_MODEL_NAME = "exp-schp-201908261155-lip.pth"

SCHP_MODEL_DIR2 = "schp"
PASCAL_MODEL_NAME = "exp-schp-201908270938-pascal-person-part.pth"


# ──────────────────────────────────────────────────────────────────────────────
# InPlaceABNSync replacement (no CUDA extension needed)
# ──────────────────────────────────────────────────────────────────────────────


class InPlaceABNSync(nn.Module):
    """Drop-in replacement for mapillary InPlaceABNSync."""

    def __init__(self, num_features, activation='leaky_relu', activation_param=0.01):
        super().__init__()
        self.activation = activation
        self.activation_param = activation_param
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x = F.batch_norm(x, self.running_mean, self.running_var,
                         self.weight, self.bias, training=self.training)
        if self.activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == 'relu':
            x = F.relu(x, inplace=True)
        return x


_BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


# ──────────────────────────────────────────────────────────────────────────────
# SCHP AugmentCE2P architecture (ResNet101 + PSP + Edge + Decoder)
# ──────────────────────────────────────────────────────────────────────────────


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1,
                 downsample=None, fist_dilation=1, multi_grid=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = _BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=dilation * multi_grid,
                               dilation=dilation * multi_grid, bias=False)
        self.bn2 = _BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = _BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=False)
        self.relu_inplace = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu_inplace(out + residual)
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size) for size in sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features,
                      kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features))

    def _make_stage(self, features, out_features, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(features, out_features, kernel_size=1, bias=False),
            InPlaceABNSync(out_features))

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(stage(feats), size=(h, w), mode='bilinear',
                                align_corners=True) for stage in self.stages] + [feats]
        return self.bottleneck(torch.cat(priors, 1))


class Edge_Module(nn.Module):
    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, 1, bias=False), InPlaceABNSync(mid_fea))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, 1, bias=False), InPlaceABNSync(mid_fea))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, 1, bias=False), InPlaceABNSync(mid_fea))
        self.conv4 = nn.Conv2d(mid_fea, out_fea, 3, padding=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, 1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        e1_fea, e1 = self.conv1(x1), self.conv4(self.conv1(x1))
        e2_fea, e2 = self.conv2(x2), self.conv4(self.conv2(x2))
        e3_fea, e3 = self.conv3(x3), self.conv4(self.conv3(x3))
        e2_fea = F.interpolate(e2_fea, size=(h, w), mode='bilinear', align_corners=True)
        e3_fea = F.interpolate(e3_fea, size=(h, w), mode='bilinear', align_corners=True)
        e2 = F.interpolate(e2, size=(h, w), mode='bilinear', align_corners=True)
        e3 = F.interpolate(e3, size=(h, w), mode='bilinear', align_corners=True)
        edge = self.conv5(torch.cat([e1, e2, e3], dim=1))
        edge_fea = torch.cat([e1_fea, e2_fea, e3_fea], dim=1)
        return edge, edge_fea


class Decoder_Module(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False), InPlaceABNSync(256))
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, 1, bias=False), InPlaceABNSync(48))
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, 1, bias=False), InPlaceABNSync(256),
            nn.Conv2d(256, 256, 1, bias=False), InPlaceABNSync(256))
        self.conv4 = nn.Conv2d(256, num_classes, 1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = self.conv3(torch.cat([xt, xl], dim=1))
        return self.conv4(x), x


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 128
        super().__init__()
        self.conv1 = _conv3x3(3, 64, stride=2)
        self.bn1 = _BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = _conv3x3(64, 64)
        self.bn2 = _BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = _conv3x3(64, 128)
        self.bn3 = _BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                        dilation=2, multi_grid=(1, 1, 1))
        self.context_encoding = PSPModule(2048, 512)
        self.edge = Edge_Module()
        self.decoder = Decoder_Module(num_classes)
        self.fushion = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False), InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, 1, bias=True))

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                _BatchNorm2d(planes * block.expansion))
        layers = []
        gen_mg = lambda i, g: g[i % len(g)] if isinstance(g, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation,
                            downsample=downsample, multi_grid=gen_mg(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=gen_mg(i, multi_grid)))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.relu3(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        x = self.context_encoding(x5)
        parsing_result, parsing_fea = self.decoder(x, x2)
        edge_result, edge_fea = self.edge(x2, x3, x4)
        x = torch.cat([parsing_fea, edge_fea], dim=1)
        fusion_result = self.fushion(x)
        return [[parsing_result, fusion_result], [edge_result]]


# ──────────────────────────────────────────────────────────────────────────────
# Affine transform utilities (from SCHP)
# ──────────────────────────────────────────────────────────────────────────────


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    return [src_point[0] * cs - src_point[1] * sn,
            src_point[0] * sn + src_point[1] * cs]


def _get_affine_transform(center, scale, rot, output_size,
                           shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    src_w = scale[0]
    dst_w, dst_h = output_size[1], output_size[0]
    src_dir = _get_dir([0, src_w * -0.5], np.pi * rot / 180)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale * shift
    src[1, :] = center + src_dir + scale * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir
    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])
    if inv:
        return cv2.getAffineTransform(np.float32(dst), np.float32(src))
    return cv2.getAffineTransform(np.float32(src), np.float32(dst))


def _transform_logits(logits, center, scale, width, height, input_size):
    trans = _get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logits.append(cv2.warpAffine(
            logits[:, :, i], trans, (int(width), int(height)),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    return np.stack(target_logits, axis=2)


def _xywh2cs(x, y, w, h, input_size):
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    aspect_ratio = input_size[1] / input_size[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    return center, np.array([w, h], dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────────────────────────────────


def _get_device():
    global _device
    if _device is None:
        import comfy.model_management
        _device = comfy.model_management.get_torch_device()
    return _device


def _load_schp_model(model_dir, model_name, num_classes, label):
    """Load a SCHP model from disk."""
    model_path = os.path.join(model_dir, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[ReFace] SCHP {label} model not found at {model_path}.\n"
            f"Please download and place as {model_path}")

    device = _get_device()
    net = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k[7:] if k.startswith("module.") else k] = v

    net.load_state_dict(new_state_dict, strict=False)
    net.to(device)
    net.eval()

    print(f"[ReFace] SCHP {label} model loaded on {device}")
    return net


def get_lip_model():
    """Get or create singleton SCHP LIP model."""
    global _lip_model
    if _lip_model is None:
        _lip_model = _load_schp_model(
            os.path.join(folder_paths.models_dir, SCHP_MODEL_DIR),
            LIP_MODEL_NAME, LIP_NUM_CLASSES, "LIP")
    return _lip_model


def get_pascal_model():
    """Get or create singleton SCHP Pascal model."""
    global _pascal_model
    if _pascal_model is None:
        _pascal_model = _load_schp_model(
            os.path.join(folder_paths.models_dir, SCHP_MODEL_DIR2),
            PASCAL_MODEL_NAME, PASCAL_NUM_CLASSES, "Pascal")
    return _pascal_model


_schp_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
])


def _run_schp(model, image_rgb, input_size, head_labels):
    """Run a single SCHP model and return head mask."""
    device = _get_device()
    h, w = image_rgb.shape[:2]

    center, scale = _xywh2cs(0, 0, w - 1, h - 1, input_size)
    trans = _get_affine_transform(center, scale, 0, input_size)
    input_img = cv2.warpAffine(
        image_rgb, trans, (input_size[1], input_size[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    tensor = _schp_transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    upsample = nn.Upsample(size=input_size, mode='bilinear', align_corners=True)
    logits = upsample(output[0][-1][0].unsqueeze(0)).squeeze(0).permute(1, 2, 0)
    logits = _transform_logits(logits.cpu().numpy(), center, scale, w, h,
                               input_size=input_size)
    parsing = np.argmax(logits, axis=2)

    mask = np.zeros((h, w), dtype=np.uint8)
    for label in head_labels:
        mask[parsing == label] = 255
    return mask


def parse_head_mask(
    image_rgb: np.ndarray,
    instance_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Parse image and return binary head mask using SCHP LIP ∪ Pascal.

    Pipeline:
      1. SCHP LIP → Hat(1) + Hair(2) + Sunglasses(4) + Face(13)
      2. SCHP Pascal → Head(1)
      3. Union of both masks
      4. Intersect with YOLO instance mask (if provided)
      5. Caller should run _fill_holes() on the result

    Args:
        image_rgb: RGB image (H, W, 3), uint8.
        instance_mask: Optional binary mask (H, W), bool or uint8.

    Returns:
        Binary mask (H, W), uint8, 0 or 255.
    """
    h, w = image_rgb.shape[:2]

    # Run both models
    lip_mask = _run_schp(get_lip_model(), image_rgb, LIP_INPUT_SIZE, LIP_HEAD_LABELS)
    pascal_mask = _run_schp(get_pascal_model(), image_rgb, PASCAL_INPUT_SIZE, PASCAL_HEAD_LABELS)

    # Union
    head_mask = np.maximum(lip_mask, pascal_mask)

    # Intersect with YOLO instance mask
    if instance_mask is not None:
        inst = instance_mask.astype(np.uint8)
        if inst.max() == 1:
            inst = inst * 255
        head_mask[inst == 0] = 0

    return head_mask


def _fill_holes(mask: np.ndarray, expand_ratio: float = 0.05) -> np.ndarray:
    """Fill holes and expand the mask outward.

    Args:
        mask: Binary mask (H, W), uint8, 0 or 255.
        expand_ratio: Expand the mask by this fraction of its bounding box diagonal.
            Default 0.05 (5%).

    Returns:
        Processed binary mask (H, W), uint8, 0 or 255.
    """
    if mask.max() == 0:
        return mask

    # Morphological closing to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Fill holes: find contours and fill
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    # Expand mask outward by expand_ratio of bbox diagonal
    if expand_ratio > 0:
        ys, xs = np.where(filled > 0)
        if len(ys) > 0:
            bbox_w = xs.max() - xs.min()
            bbox_h = ys.max() - ys.min()
            diag = np.sqrt(bbox_w ** 2 + bbox_h ** 2)
            expand_px = max(1, int(diag * expand_ratio))
            dilate_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (expand_px * 2 + 1, expand_px * 2 + 1))
            filled = cv2.dilate(filled, dilate_kernel, iterations=1)

    return filled
