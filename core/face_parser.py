"""BiSeNet face parsing for head segmentation.

Uses the face-parsing BiSeNet model to obtain a full head mask.
Model: 79999_iter.pth from https://github.com/zllrunning/face-parsing.PyTorch

BiSeNet label map:
  0: background, 1: skin, 2: l_brow, 3: r_brow, 4: l_eye, 5: r_eye,
  6: eye_g (glasses), 7: l_ear, 8: r_ear, 9: ear_r (earring), 10: nose,
  11: mouth, 12: u_lip, 13: l_lip, 14: neck, 15: neck_l (necklace),
  16: cloth, 17: hair, 18: hat
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

import folder_paths

# Labels that belong to the head (everything except background, neck, necklace, cloth)
HEAD_LABELS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18}

_parser_model = None
_device = None

BISENET_MODEL_NAME = "bisenet_face_parsing.pth"
BISENET_MODEL_URL = "https://github.com/zllrunning/face-parsing.PyTorch/releases/download/v1.0/79999_iter.pth"


# ──────────────────────────────────────────────────────────────────────────────
# BiSeNet architecture — verbatim from face-parsing.PyTorch
# https://github.com/zllrunning/face-parsing.PyTorch
# ──────────────────────────────────────────────────────────────────────────────


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super().__init__()
        self.conv1 = _conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = _conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
            )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def _create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for _ in range(bnum - 1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = _create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = _create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = _create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = _create_layer_basic(256, 512, bnum=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        x = self.layer1(x)
        feat8 = self.layer2(x)
        feat16 = self.layer3(feat8)
        feat32 = self.layer4(feat16)
        return feat8, feat16, feat32


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_chan, out_chan, kernel_size=ks, stride=stride,
                              padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_chan)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out


class ContextPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

    def forward(self, x):
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out


class BiSeNet(nn.Module):
    def __init__(self, n_classes=19):
        super().__init__()
        self.cp = ContextPath()
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)
        feat_sp = feat_res8
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)
        return feat_out, feat_out16, feat_out32


# ──────────────────────────────────────────────────────────────────────────────
# Model loading & inference
# ──────────────────────────────────────────────────────────────────────────────


def _get_model_dir():
    return os.path.join(folder_paths.models_dir, "reface")


def _download_bisenet():
    """Download BiSeNet model if not present."""
    model_dir = _get_model_dir()
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, BISENET_MODEL_NAME)

    if os.path.isfile(model_path):
        return model_path

    print(f"[ReFace] Downloading BiSeNet face parsing model to {model_path} ...")
    import urllib.request
    urllib.request.urlretrieve(BISENET_MODEL_URL, model_path)
    print("[ReFace] BiSeNet model downloaded.")
    return model_path


def _get_device():
    global _device
    if _device is None:
        import comfy.model_management
        _device = comfy.model_management.get_torch_device()
    return _device


def get_face_parser():
    """Get or create singleton BiSeNet face parser."""
    global _parser_model
    if _parser_model is not None:
        return _parser_model

    model_path = _download_bisenet()
    device = _get_device()

    net = BiSeNet(n_classes=19)
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Strip "module." prefix if the model was saved with DataParallel
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()

    _parser_model = net
    return _parser_model


_to_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def parse_head_mask(
    image_rgb: np.ndarray,
    instance_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Parse image and return binary head mask, optionally constrained to a human instance.

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

    # If instance_mask is given, crop the image to the instance bbox
    # and run BiSeNet on the cropped region for better accuracy.
    if instance_mask is not None:
        inst = instance_mask.astype(np.uint8)
        if inst.max() == 1:
            inst = inst * 255
        ys, xs = np.where(inst > 0)
        if len(ys) == 0:
            return np.zeros((h, w), dtype=np.uint8)
        crop_y1, crop_y2 = int(ys.min()), int(ys.max()) + 1
        crop_x1, crop_x2 = int(xs.min()), int(xs.max()) + 1
        cropped_img = image_rgb[crop_y1:crop_y2, crop_x1:crop_x2]
        cropped_inst = inst[crop_y1:crop_y2, crop_x1:crop_x2]
    else:
        cropped_img = image_rgb
        cropped_inst = None
        crop_y1, crop_x1 = 0, 0

    ch, cw = cropped_img.shape[:2]

    # BiSeNet expects 512x512 input
    from PIL import Image as PILImage
    pil_img = PILImage.fromarray(cropped_img).resize((512, 512), PILImage.BILINEAR)
    tensor = _to_tensor(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(tensor)[0]  # Take main output
    parsing = out.squeeze(0).cpu().numpy().argmax(0)  # (512, 512)

    # Build head mask from relevant labels
    head_mask_small = np.zeros_like(parsing, dtype=np.uint8)
    for label in HEAD_LABELS:
        head_mask_small[parsing == label] = 255

    # Resize back to cropped region size
    mask_pil = PILImage.fromarray(head_mask_small).resize((cw, ch), PILImage.NEAREST)
    head_mask_crop = np.array(mask_pil)

    # Intersect with instance mask if provided
    if cropped_inst is not None:
        head_mask_crop[cropped_inst == 0] = 0

    # Place cropped result back into full-size mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[crop_y1:crop_y1 + ch, crop_x1:crop_x1 + cw] = head_mask_crop

    return full_mask


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes inside the mask using contour-based approach."""
    import cv2

    # Morphological closing to connect nearby regions
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Fill holes: find contours and fill
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, cv2.FILLED)

    return filled
