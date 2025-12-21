from typing import Dict
import numpy as np
import torch
from utils import heatmap_nms, transpose_and_gather_feat, gather_feat, topk

from datasets._coco_constants import COCO_IDS


def decode_dets(
    tl_hmap: torch.Tensor,
    br_hmap: torch.Tensor,
    ct_hmap: torch.Tensor,
    tl_regs: torch.Tensor,
    br_regs: torch.Tensor,
    ct_regs: torch.Tensor,
    tl_embd: torch.Tensor,
    br_embd: torch.Tensor,
    topk_K: int,
    nms_kernel: int,
    ae_threshold: float,
    num_dets: int = 1000,
):
    B, C, H, W = tl_hmap.shape

    tl_hmap = torch.sigmoid(tl_hmap)
    br_hmap = torch.sigmoid(br_hmap)
    ct_hmap = torch.sigmoid(ct_hmap)

    # perform nms on the heatmaps
    tl_hmap = heatmap_nms(heatmap=tl_hmap, kernel=nms_kernel)
    br_hmap = heatmap_nms(heatmap=br_hmap, kernel=nms_kernel)
    ct_hmap = heatmap_nms(heatmap=ct_hmap, kernel=nms_kernel)

    tl_scores, tl_inds, tl_classes, tl_ys, tl_xs = topk(tl_hmap, k=topk_K)
    br_scores, br_inds, br_classes, br_ys, br_xs = topk(br_hmap, k=topk_K)
    ct_scores, ct_inds, ct_classes, ct_ys, ct_xs = topk(ct_hmap, k=topk_K)

    tl_xs = tl_xs.view(B, topk_K, 1).expand(B, topk_K, topk_K)
    br_xs = br_xs.view(B, 1, topk_K).expand(B, topk_K, topk_K)

    tl_ys = tl_ys.view(B, topk_K, 1).expand(B, topk_K, topk_K)
    br_ys = br_ys.view(B, 1, topk_K).expand(B, topk_K, topk_K)

    tl_regs = transpose_and_gather_feat(tl_regs, tl_inds)
    tl_embd = transpose_and_gather_feat(tl_embd, tl_inds)
    br_regs = transpose_and_gather_feat(br_regs, br_inds)
    br_embd = transpose_and_gather_feat(br_embd, br_inds)
    ct_regs = transpose_and_gather_feat(ct_regs, ct_inds)

    tl_regs = tl_regs.view(B, topk_K, 1, 2)
    br_regs = br_regs.view(B, 1, topk_K, 2)

    tl_xs = tl_xs + tl_regs[..., 0]
    tl_ys = tl_ys + tl_regs[..., 1]
    br_xs = br_xs + br_regs[..., 0]
    br_ys = br_ys + br_regs[..., 1]
    ct_xs = ct_xs + ct_regs[..., 0]
    ct_ys = ct_ys + ct_regs[..., 1]

    # all possible boxes based on topk corners (ignoring class)
    bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)  # [B, K, K, 4]

    tl_embd = tl_embd.view(B, topk_K, 1).expand(B, topk_K, topk_K)
    br_embd = br_embd.view(B, 1, topk_K).expand(B, topk_K, topk_K)

    # distance matrix of B x K x K
    dists = torch.abs(tl_embd - br_embd)

    tl_scores = tl_scores.view(B, topk_K, 1).expand(B, topk_K, topk_K)
    br_scores = br_scores.view(B, 1, topk_K).expand(B, topk_K, topk_K)

    # score matrix of B x K x K
    score_matrix = (tl_scores + br_scores) / 2

    # reject boxes based on classes
    tl_classes = tl_classes.view(B, topk_K, 1).expand(B, topk_K, topk_K)
    br_classes = br_classes.view(B, 1, topk_K).expand(B, topk_K, topk_K)
    cls_inds = tl_classes != br_classes

    # reject boxes based on associative embedding distance
    dist_inds = dists > ae_threshold

    # reject boxes based on invalid shape
    width_inds = br_xs < tl_xs
    height_inds = br_ys < tl_ys

    score_matrix[cls_inds] = -1
    score_matrix[dist_inds] = -1
    score_matrix[width_inds] = -1
    score_matrix[height_inds] = -1

    # flatten the score matrix
    scores = score_matrix.view(B, -1)
    scores, inds = torch.topk(scores, num_dets)
    scores = scores[:, :, None]  # [B, num_dets, 1]

    bboxes = bboxes.view(B, -1, 4)  # [B, K * K, 4]
    bboxes = gather_feat(bboxes, inds)  # [B, num_dets, 4]

    # since at this point tl_classes and br_classes are same
    # we can use either of them to represent actual classes.
    tl_classes = tl_classes.contiguous().view(B, -1, 1)  # [B, K * K, 1]
    classes = gather_feat(tl_classes, inds)  # [B, num_dets, 1]

    tl_scores = tl_scores.contiguous().view(B, -1, 1)  # [B, K * K, 1]
    br_scores = br_scores.contiguous().view(B, -1, 1)  # [B, K * K, 1]
    tl_scores = gather_feat(tl_scores, inds)
    br_scores = gather_feat(br_scores, inds)

    # [B, K, 4]
    centers = torch.stack((ct_xs, ct_ys, ct_classes.float(), ct_scores), dim=-1)
    return {
        "bboxes": bboxes,
        "classes": classes,
        "scores": scores,
        "tl_scores": tl_scores,
        "br_scores": br_scores,
        "centers": centers,
    }


def rescale_dets(
    bboxes: torch.Tensor,
    scores: torch.Tensor,
    centers: torch.Tensor,
    ratios: torch.Tensor,
    borders: torch.Tensor,
    original_size: torch.Tensor,
):
    xs, ys = bboxes[..., 0:4:2], bboxes[..., 1:4:2]
    xs /= ratios[:, 1][:, None, None]
    ys /= ratios[:, 0][:, None, None]
    # subract top-left coordinate of border
    xs -= borders[:, 2][:, None, None]
    ys -= borders[:, 0][:, None, None]

    # This is some unknown hack that author of centernet have put
    # I think it's mostly with respect to clipping but looks like
    # some hack to improve the AP Need to verify by removing ths.

    tx_inds = xs[:, :, 0] <= -5
    bx_inds = xs[:, :, 1] >= original_size[:, 1] + 5
    ty_inds = ys[:, :, 0] <= -5
    by_inds = ys[:, :, 1] >= original_size[:, 0] + 5

    scores[:, tx_inds[0, :]] = -1
    scores[:, bx_inds[0, :]] = -1
    scores[:, ty_inds[0, :]] = -1
    scores[:, by_inds[0, :]] = -1

    ## Now we have the xs and ys in original image space

    xs = xs.clamp(min=0.0).clamp(max=original_size[:, 1][:, None, None])
    ys = ys.clamp(min=0.0).clamp(max=original_size[:, 0][:, None, None])

    # center scaling
    centers[..., 0] /= ratios[:, 1][:, None]
    centers[..., 0] /= ratios[:, 0][:, None]
    centers[..., 0] -= borders[:, 2][:, None]
    centers[..., 1] -= borders[:, 0][:, None]

    centers[..., 0] = centers[..., 0].clamp(min=0.0).clamp(max=original_size[:, 1])
    centers[..., 1] = centers[..., 1].clamp(min=0.0).clamp(max=original_size[:, 0])

    out = bboxes.clone()
    out[..., 0:4:2] = xs
    out[..., 1:4:2] = ys

    return out, scores, centers


@torch.compile()
def _get_center_region(tlx, tly, brx, bry, n):

    a = (n + 1) / (2.0 * n)
    b = (n - 1) / (2.0 * n)

    ctlx = a * tlx + b * brx
    ctly = a * tly + b * bry
    cbrx = b * tlx + a * brx
    cbry = b * tly + a * bry

    ctlx = ctlx[None, :]
    ctly = ctly[None, :]
    cbrx = cbrx[None, :]
    cbry = cbry[None, :]

    return ctlx, ctly, cbrx, cbry


def _do_center_match(dets: torch.Tensor, centers: torch.Tensor, n: int):

    ctlx, ctly, cbrx, cbry = _get_center_region(
        tlx=dets[:, 0], tly=dets[:, 1], brx=dets[:, 2], bry=dets[:, 3], n=n
    )
    det_classes = dets[:, -1][None, :]  # [1, num_dets]

    center_x = centers[:, 0][:, None]  # [topk_K, 1]
    center_y = centers[:, 1][:, None]  # [topk_K, 1]
    center_classes = centers[:, 2][:, None]  # [topk_K, 1]

    # (topk_K, num_dets) matrix
    valid_mask = (
        (center_x > ctlx)
        & (center_y > ctly)
        & (center_x < cbrx)
        & (center_y < cbry)
        & (center_classes == det_classes)
    )

    matching_box_ind = valid_mask.any(dim=0)
    if matching_box_ind.any():
        matching_center_ind = torch.argmax(
            valid_mask[:, matching_box_ind].float(), dim=0
        )

        # this normalizes to (tl_score + br_score + ct_score) / 3
        dets[matching_box_ind, 4] = (
            2 * dets[matching_box_ind, 4] + centers[matching_center_ind, 3]
        ) / 3

    dets[~matching_box_ind, 4] = -1

    return dets


def center_match(detections: torch.Tensor, centers: torch.Tensor):
    valid_inds = detections[:, 4] > -1
    valid_detections = detections[valid_inds]

    box_width = valid_detections[:, 2] - valid_detections[:, 0]
    box_height = valid_detections[:, 3] - valid_detections[:, 1]

    small_box_ind = box_width * box_height <= 22500
    large_box_ind = box_width * box_height > 22500

    small_dets = valid_detections[small_box_ind]
    large_dets = valid_detections[large_box_ind]

    SMALL_N, LARGE_N = 3, 5

    small_dets = _do_center_match(small_dets, centers, n=SMALL_N)
    large_dets = _do_center_match(large_dets, centers, n=LARGE_N)

    detections = torch.concatenate((small_dets, large_dets), dim=0)
    detections = detections[torch.argsort(-detections[:, 4])]
    clses = detections[..., -1]
    return detections, clses


def convert_to_coco_eval_format(all_bboxes: Dict[str, Dict[int, np.ndarray]]):
    detections = []
    for image_id in all_bboxes:
        for cls_ind in all_bboxes[image_id]:
            category_id = COCO_IDS[cls_ind - 1]
            for bbox in all_bboxes[image_id][cls_ind]:
                # xyxy to xywh
                bbox[2] -= bbox[0]
                bbox[3] -= bbox[1]
                score = bbox[4]

                bbox_out = list(map(lambda x: float("{:.2f}".format(x)), bbox[0:4]))
                detection = {
                    "image_id": int(image_id),
                    "category_id": int(category_id),
                    "bbox": bbox_out,
                    "score": float("{:.2f}".format(score)),
                }
                detections.append(detection)

    return detections
