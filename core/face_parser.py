"""SCHP (Self-Correction Human Parsing) for head segmentation.

Uses the SCHP model with Pascal-Person-Part dataset for full-body head parsing.
Model: exp-schp-201908270938-pascal-person-part.pth
From: https://github.com/GoGoDuck912/Self-Correction-Human-Parsing

Pascal-Person-Part label map:
  0: Background, 1: Head, 2: Torso, 3: Upper Arms,
  4: Lower Arms, 5: Upper Legs, 6: Lower Legs
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from collections import OrderedDict

import folder_paths

# Head label in Pascal-Person-Part
HEAD_LABEL = 1

SCHP_INPUT_SIZE = [512, 512]

_parser_model = None
_device = None

SCHP_MODEL_DIR = "schp"
SCHP_MODEL_NAME = "exp-schp-201908270938-pascal-person-part.pth"


# ──────────────────────────────────────────────────────────────────────────────
# InPlaceABNSync replacement (no CUDA extension needed)
# Compatible state_dict: weight, bias, running_mean, running_var, num_batches_tracked
# ──────────────────────────────────────────────────────────────────────────────


class InPlaceABNSync(nn.Module):
    """Drop-in replacement for mapillary InPlaceABNSync.

    Combines BatchNorm + optional activation (leaky_relu or none).
    State dict keys match the original exactly.
    """

    def __init__(self, num_features, activation='leaky_relu', activation_param=0.01):
        super().__init__()
        self.num_features = num_features
        self.activation = activation
        self.activation_param = activation_param
        # BN parameters stored as direct attributes (matching original state dict)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

    def forward(self, x):
        x = F.batch_norm(
            x, self.running_mean, self.running_var,
            self.weight, self.bias, training=self.training,
        )
        if self.activation == 'leaky_relu':
            x = F.leaky_relu(x, negative_slope=self.activation_param, inplace=True)
        elif self.activation == 'relu':
            x = F.relu(x, inplace=True)
        # activation == 'none' → no activation
        return x


import functools
_BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')


# ──────────────────────────────────────────────────────────────────────────────
# SCHP AugmentCE2P architecture (ResNet101 + PSP + Edge + Decoder)
# Verbatim from Self-Correction-Human-Parsing/networks/AugmentCE2P.py
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
        self.dilation = dilation
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out = out + residual
        out = self.relu_inplace(out)
        return out


class PSPModule(nn.Module):
    def __init__(self, features, out_features=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList(
            [self._make_stage(features, out_features, size) for size in sizes]
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features + len(sizes) * out_features, out_features,
                      kernel_size=3, padding=1, dilation=1, bias=False),
            InPlaceABNSync(out_features),
        )

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, out_features, kernel_size=1, bias=False)
        bn = InPlaceABNSync(out_features)
        return nn.Sequential(prior, conv, bn)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [
            F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True)
            for stage in self.stages
        ] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return bottle


class Edge_Module(nn.Module):
    def __init__(self, in_fea=[256, 512, 1024], mid_fea=256, out_fea=2):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_fea[0], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_fea[1], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_fea[2], mid_fea, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(mid_fea),
        )
        self.conv4 = nn.Conv2d(mid_fea, out_fea, kernel_size=3, padding=1, dilation=1, bias=True)
        self.conv5 = nn.Conv2d(out_fea * 3, out_fea, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, x1, x2, x3):
        _, _, h, w = x1.size()
        edge1_fea = self.conv1(x1)
        edge1 = self.conv4(edge1_fea)
        edge2_fea = self.conv2(x2)
        edge2 = self.conv4(edge2_fea)
        edge3_fea = self.conv3(x3)
        edge3 = self.conv4(edge3_fea)

        edge2_fea = F.interpolate(edge2_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge3_fea = F.interpolate(edge3_fea, size=(h, w), mode='bilinear', align_corners=True)
        edge2 = F.interpolate(edge2, size=(h, w), mode='bilinear', align_corners=True)
        edge3 = F.interpolate(edge3, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge1, edge2, edge3], dim=1)
        edge_fea = torch.cat([edge1_fea, edge2_fea, edge3_fea], dim=1)
        edge = self.conv5(edge)
        return edge, edge_fea


class Decoder_Module(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(48),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
        )
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xl):
        _, _, h, w = xl.size()
        xt = F.interpolate(self.conv1(xt), size=(h, w), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x = self.conv3(x)
        seg = self.conv4(x)
        return seg, x


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
            nn.Conv2d(1024, 256, kernel_size=1, padding=0, dilation=1, bias=False),
            InPlaceABNSync(256),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True),
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, multi_grid=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                _BatchNorm2d(planes * block.expansion),
            )
        layers = []
        generate_multi_grid = lambda index, grids: grids[index % len(grids)] if isinstance(grids, tuple) else 1
        layers.append(block(self.inplanes, planes, stride, dilation=dilation,
                            downsample=downsample,
                            multi_grid=generate_multi_grid(0, multi_grid)))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation,
                                multi_grid=generate_multi_grid(i, multi_grid)))
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


def _build_schp_model(num_classes=7):
    """Build SCHP ResNet101 model for Pascal-Person-Part."""
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


# ──────────────────────────────────────────────────────────────────────────────
# Affine transform utilities (from SCHP)
# ──────────────────────────────────────────────────────────────────────────────


def _get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def _get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs
    return src_result


def _get_affine_transform(center, scale, rot, output_size, shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])
    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[1]
    dst_h = output_size[0]

    rot_rad = np.pi * rot / 180
    src_dir = _get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, (dst_w - 1) * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]
    dst[1, :] = np.array([(dst_w - 1) * 0.5, (dst_h - 1) * 0.5]) + dst_dir

    src[2:, :] = _get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def _transform_logits(logits, center, scale, width, height, input_size):
    trans = _get_affine_transform(center, scale, 0, input_size, inv=1)
    channel = logits.shape[2]
    target_logits = []
    for i in range(channel):
        target_logit = cv2.warpAffine(
            logits[:, :, i], trans, (int(width), int(height)),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        target_logits.append(target_logit)
    return np.stack(target_logits, axis=2)


def _xywh2cs(x, y, w, h, input_size):
    """Convert xywh bbox to center/scale for affine transform."""
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)
    aspect_ratio = input_size[1] / input_size[0]
    if w > aspect_ratio * h:
        h = w / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array([w, h], dtype=np.float32)
    return center, scale


# ──────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────────────────────────────────


def _get_model_dir():
    return os.path.join(folder_paths.models_dir, "reface")


def _get_device():
    global _device
    if _device is None:
        import comfy.model_management
        _device = comfy.model_management.get_torch_device()
    return _device


def get_face_parser():
    """Get or create singleton SCHP face parser."""
    global _parser_model
    if _parser_model is not None:
        return _parser_model

    model_dir = os.path.join(folder_paths.models_dir, SCHP_MODEL_DIR)
    model_path = os.path.join(model_dir, SCHP_MODEL_NAME)

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"[ReFace] SCHP Pascal-Person-Part model not found at {model_path}.\n"
            f"Please download it from:\n"
            f"  https://drive.google.com/file/d/1E5YwNKW2VOEayK9mWCS3Kpsxf-3z04ZE/view\n"
            f"Place as {model_path}"
        )

    device = _get_device()

    net = _build_schp_model(num_classes=7)

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    # Strip "module." prefix (DataParallel)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    net.to(device)
    net.eval()

    _parser_model = net
    print(f"[ReFace] SCHP Pascal-Person-Part model loaded on {device}")
    return _parser_model


# SCHP uses BGR mean/std normalization
_schp_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229]),
])


def parse_head_mask(
    image_rgb: np.ndarray,
    instance_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Parse image and return binary head mask using SCHP.

    Args:
        image_rgb: RGB image as numpy array (H, W, 3), uint8.
        instance_mask: Optional binary mask (H, W), bool or uint8.
            If provided, the head parsing result is intersected with this mask
            so that only the head of the specified person instance is returned.

    Returns:
        Binary mask as numpy array (H, W), uint8, 0 or 255.
    """
    model = get_face_parser()
    device = _get_device()
    h, w = image_rgb.shape[:2]

    # Compute center/scale for the full image
    center, scale = _xywh2cs(0, 0, w - 1, h - 1, SCHP_INPUT_SIZE)

    # Build affine transform to warp image to SCHP input size
    trans = _get_affine_transform(center, scale, 0, SCHP_INPUT_SIZE)
    input_img = cv2.warpAffine(
        image_rgb, trans, (SCHP_INPUT_SIZE[1], SCHP_INPUT_SIZE[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
    )

    # Normalize and run model
    tensor = _schp_transform(input_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)

    # Take fusion result (output[0][-1]), upsample to input size
    upsample = nn.Upsample(size=SCHP_INPUT_SIZE, mode='bilinear', align_corners=True)
    upsample_output = upsample(output[0][-1][0].unsqueeze(0))
    upsample_output = upsample_output.squeeze(0)
    upsample_output = upsample_output.permute(1, 2, 0)  # CHW -> HWC

    # Inverse affine transform back to original image size
    logits = _transform_logits(
        upsample_output.cpu().numpy(), center, scale, w, h, input_size=SCHP_INPUT_SIZE,
    )
    parsing = np.argmax(logits, axis=2)

    # Head mask: label == 1
    head_mask = np.where(parsing == HEAD_LABEL, 255, 0).astype(np.uint8)

    # Intersect with instance mask if provided
    if instance_mask is not None:
        inst = instance_mask.astype(np.uint8)
        if inst.max() == 1:
            inst = inst * 255
        head_mask[inst == 0] = 0

    return head_mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside the mask using contour-based approach."""
    # Morphological closing to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Fill holes: find contours and fill
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    return filled
