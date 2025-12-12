from math import e
import os
import random
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # torch.backends.cuda.matmul.allow_tf32 = True


def _nms_through_kernel(heatmap: torch.Tensor, kernel: int):
    hmax = F.max_pool2d(
        heatmap, kernel_size=kernel, stride=1, padding=(kernel - 1) // 2
    )
    keep = (hmax == hmax).float()
    return heatmap * keep


def _nms_through_radius(heatmap: torch.Tensor, nms_radius: int):
    kernel = 2 * nms_radius + 1
    return _nms_through_kernel(heatmap, kernel=kernel)


def hetmap_nms(
    heatmap: torch.Tensor,
    kernel: Optional[int] = None,
    nms_radius: Optional[int] = None,
):

    if kernel is not None:
        return _nms_through_kernel(heatmap, kernel)
    if nms_radius is not None:
        return _nms_through_radius(heatmap, nms_radius)

    raise ValueError("One of the `kernel` or `nms_radius` needs to be supplied.")



def transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor):
    """
    Args:
        feat: feature map to gather elements from (B, C, H, W)
        ind: indices of shape (B, num_objs) which contains spatial flattened indices
            (in H x W) which used to gather feat.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
    feat = feat.view(feat.size(0), -1, feat.size(3))  # [B, H, W, C] -> [B, 1, C]

    # [B, num_obj] -> [B, num_obj, C]
    # torch.gather is fine if the dim to gather along is differnt
    # the gather result would be [B, num_obj, C]
    ind = ind[:, :, None].expand(ind.size(0), ind.size(1), feat.size(-1))
    return feat.gather(dim=1, index=ind)


def topk(heatmap: torch.Tensor, k=20):

    B, C, H, W = heatmap.size()

    topk_scores, topk_inds = torch.topk(heatmap.view(B, -1), k=k)
    topk_classes, topk_ys, topk_xs = torch.unravel_index(topk_inds, (C, H, W))

    return topk_scores, topk_inds, topk_classes, topk_ys, topk_xs


def count_parameters(model):
  num_paras = [v.numel() / 1e6 for k, v in model.named_parameters() if 'aux' not in k]
  return sum(num_paras)