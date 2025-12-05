import torch
import torch.nn as nn
import numpy as np


def _gather_feat(feat: torch.Tensor, ind: torch.Tensor, mask=None):
    """
    Gathers features from a flattened feature map based on topk indices,
    with optional masking to filter valid detections.

    Args:
        feat: Flattened feature map with shape (B, H*W, C), where features
              are already arranged in a flat spatial layout
        ind: Topk indices with shape (B, num_obj), where each index is in range
             [0, H*W) representing spatial positions to gather
        mask: Optional binary mask with shape (B, num_obj) indicating valid detections.
              True/1 marks valid detections, False/0 marks padding.

    Returns:
        If mask is None: Gathered features with shape (B, num_obj, C)
        If mask is provided: Filtered features with shape (N, C), where N is
                            the total number of valid (non-masked) detections
                            across all batches
    """
    dim = feat.size(2)  # C: number of channels
    ind = ind.unsqueeze(2).expand(
        ind.size(0), ind.size(1), dim
    )  # (B, num_obj) -> (B, num_obj, C)
    feat = feat.gather(
        1, ind
    )  # Gather along spatial dim: (B, H*W, C) -> (B, num_obj, C)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)  # (B, num_obj) -> (B, num_obj, C)
        feat = feat[mask]  # Boolean indexing flattens: (B, num_obj, C) -> (N*C,)
        feat = feat.view(-1, dim)  # Reshape to (N, C)
    return feat


def _transpose_and_gather_feat(feat: torch.Tensor, ind: torch.Tensor):
    """
    Gathers spatial features from feature map based on flattened spatial indices
    for each Batch and Object.

    Args:
        feat: Feature map with shape (B, C, H, W)
        ind: Flattened spatial indices with shape (B, num_obj), where each
             index is in range [0, H*W) representing positions in the
             flattened spatial dimension

    Returns:
        Gathered features with shape (B, num_obj, C)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()  # (B, C, H, W) -> (B, H, W, C)
    feat = feat.view(feat.size(0), -1, feat.size(3))  # (B, H, W, C) -> (B, H * W, C)
    ind = ind[:, :, None].expand(
        ind.size(0), ind.size(1), feat.shape[-1]
    )  # (B, num_obj) -> (B, num_obj, C)
    return feat.gather(
        1, ind
    )  # (B, H * W, C) gather along H * W to get (B, num_obj, C)


def _nms(heat: torch.Tensor, kernel=1):
    pad = (kernel - 1) // 2

    hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def _topk(scores, K=20):
    "Scores here is of size B, C, H, W"
    batch, cat, height, width = scores.size()

    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), K)
    topk_clses = (topk_inds / (height * width)).int()

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def _decode(
    tl_heat,
    br_heat,
    tl_tag,
    br_tag,
    tl_regr,
    br_regr,
    K=100,
    kernel=1,
    ae_threshold=1,
    num_dets=1000,
):
    """
    tl_heat: (B, C, H, W) - top-left corner heatmap
    br_heat: (B, C, H, W) - bottom-right corner heatmap
    tl_tag:  (B, 1, H, W) - top-left corner embedding
    br_tag:  (B, 1, H, W) - bottom-right corner embedding
    tl_regr: (B, 2, H, W) - top-left corner regression
    br_regr: (B, 2, H, W) - bottom-right corner regression
    --------------
    returns:
    detections: (B, num_dets, 6) - [x1, y1, x2, y2, score, class]
    --------------
    """
    batch, cat, height, width = tl_heat.size()

    tl_heat = torch.sigmoid(tl_heat)
    br_heat = torch.sigmoid(br_heat)

    # nms on heatmaps
    tl_heat = _nms(tl_heat, kernel=kernel)
    br_heat = _nms(br_heat, kernel=kernel)

    tl_scores, tl_inds, tl_classes, tl_ys, tl_xs = _topk(tl_heat, K=K)
    br_scores, br_inds, br_classes, br_ys, br_xs = _topk(br_heat, K=K)

    tl_ys = tl_ys.view(batch, K, 1).expand(batch, K, K)
    tl_xs = tl_xs.view(batch, K, 1).expand(batch, K, K)
    br_ys = br_ys.view(batch, 1, K).expand(batch, K, K)
    br_xs = br_xs.view(batch, 1, K).expand(batch, K, K)

    if tl_regr is not None and br_regr is not None:
        tl_regr = _transpose_and_gather_feat(tl_regr, tl_inds)
        br_regr = _transpose_and_gather_feat(br_regr, br_inds)
        tl_regr = tl_regr.view(batch, K, 1, 2)
        br_regr = br_regr.view(batch, 1, K, 2)

        tl_xs = tl_xs + tl_regr[..., 0]
        tl_ys = tl_ys + tl_regr[..., 1]
        br_xs = br_xs + br_regr[..., 0]
        br_ys = br_ys + br_regr[..., 1]

    # all possible boxes based on top k corners (ignoring class)
    # Each box is [tl_x, tl_y, br_x, br_y]
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)  # (B, K, K, 4)

    tl_tag = _transpose_and_gather_feat(tl_tag, tl_inds)
    br_tag = _transpose_and_gather_feat(br_tag, br_inds)
    tl_tag = tl_tag.view(batch, K, 1)
    br_tag = br_tag.view(batch, 1, K)

    dists = torch.abs(tl_tag - br_tag)

    # Initial confidence score for each box based on average of corner scores
    tl_scores = tl_scores.view(batch, K, 1).expand(batch, K, K)
    br_scores = br_scores.view(batch, K, 1).expand(batch, K, K)
    scores = (tl_scores + br_scores) / 2

    # reject boxes based on classes meaning corners must belong to same class
    tl_classes = tl_classes.view(batch, K, 1).expand(batch, K, K)
    br_classes = br_classes.view(batch, K, 1).expand(batch, K, K)
    cls_inds = tl_classes != br_classes

    # reject boxes based on distances
    dist_inds = dists > ae_threshold

    # reject boxes based on width and height
    width_inds = br_xs < tl_xs
    height_inds = br_ys < tl_ys

    scores[cls_inds] = -1
    scores[dist_inds] = -1
    scores[width_inds] = -1
    scores[height_inds] = -1

    scores = scores.view(batch, -1)  # (B, K * K)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores.unsqueeze(2)

    bboxes = bboxes.view(batch, -1, 4)  # (B, K * K, 4)
    bboxes = _gather_feat(bboxes, inds)

    clses = tl_classes.contiguous().view(batch, -1, 1)  # (B, K * K, 1)
    clses = _gather_feat(clses, inds).float()

    tl_scores = tl_scores.contiguous().view(batch, -1, 1)  # (B, K * K, 1)
    tl_scores = _gather_feat(tl_scores, inds).float()
    br_scores = br_scores.contiguous().view(batch, -1, 1)  # (B, K * K, 1)
    br_scores = _gather_feat(br_scores, inds).float()

    detections = torch.cat([bboxes, scores, tl_scores, br_scores, clses], dim=2)
    return detections


def _rescale_dets(detections, ratios, borders, sizes):
    xs, ys = detections[..., 0:4:2], detections[..., 1:4:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]
    np.clip(xs, 0, sizes[:, 1][:, None, None], out=xs)
    np.clip(ys, 0, sizes[:, 0][:, None, None], out=ys)
