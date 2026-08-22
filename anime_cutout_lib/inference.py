"""轻量级推理封装，基于 ISNet(ISNetDIS) 的动漫人物抠图。

Derived from SkyTNT/anime-segmentation (Apache License 2.0):
https://github.com/SkyTNT/anime-segmentation

Modifications vs upstream inference.py:
- 移除 pytorch_lightning 等训练期依赖，仅保留推理所需最小实现；
- 改为加载 plain state_dict 权重（`isnetis.ckpt`）的轻量封装；
- 接口适配插件调用（load_model / get_mask / render_result）。

Licensed under the Apache License, Version 2.0; see LICENSE in this
directory.
"""

import torch
import torch.nn as nn
import numpy as np
import cv2

from .isnet import ISNetDIS, ISNetGTEncoder


class AnimeSegModel(nn.Module):
    """与官方 train.py 中 AnimeSegmentation 结构一致（仅保留推理所需部分）。"""

    def __init__(self):
        super().__init__()
        self.net = ISNetDIS()
        self.gt_encoder = ISNetGTEncoder()
        self.gt_encoder.requires_grad_(False)

    def forward(self, x):
        return self.net(x)[0][0].sigmoid()


def load_model(ckpt_path: str, device: str = "cpu") -> AnimeSegModel:
    """从 ckpt 加载模型并移动到指定设备。"""
    model = AnimeSegModel()
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    model.device = torch.device(device)
    return model


def get_mask(model: AnimeSegModel, input_img: np.ndarray, use_amp: bool = True, s: int = 1024) -> np.ndarray:
    """输入 RGB ndarray (H, W, 3)，返回 mask (H, W, 1) float 0~1。"""
    input_img = (input_img / 255).astype(np.float32)
    h, w = h0, w0 = input_img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(input_img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    tmp_img = torch.from_numpy(img_input).type(torch.FloatTensor).to(model.device)
    with torch.no_grad():
        if use_amp and model.device.type == "cuda":
            with torch.autocast(device_type="cuda"):
                pred = model(tmp_img)
            pred = pred.to(dtype=torch.float32)
        else:
            pred = model(tmp_img)
        pred = pred.cpu().numpy()[0]
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
        pred = cv2.resize(pred, (w0, h0))[:, :, np.newaxis]
    return pred


def render_result(original_bgr: np.ndarray, mask: np.ndarray, mode: str) -> np.ndarray:
    """根据模式渲染结果。

    Args:
        original_bgr: BGR 原图 (H, W, 3)。
        mask: (H, W, 1) float 0~1。
        mode: transparent / white / mask / compare。

    Returns:
        处理后的图像。transparent 模式返回 BGRA，其余返回 BGR。
    """
    img = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)
    if mode == "mask":
        gray = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if mode == "white":
        white = (mask * img + 255 * (1 - mask)).astype(np.uint8)
        return cv2.cvtColor(white, cv2.COLOR_RGB2BGR)
    if mode == "compare":
        matted = (mask * img + (1 - mask)).astype(np.uint8)
        mask3 = (np.clip(mask, 0, 1).repeat(3, axis=2) * 255).astype(np.uint8)
        merged = np.concatenate((img, matted, mask3), axis=1)
        return cv2.cvtColor(merged, cv2.COLOR_RGB2BGR)
    # transparent
    alpha = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
    rgba = np.concatenate((img, alpha), axis=2).astype(np.uint8)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
