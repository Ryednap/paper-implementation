from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import transpose_and_gather_feat

_FOCAL_ALPHA = 2
_FOCAL_BETA = 4

_EMBD_DELTA = 1


def _neg_loss(preds: List[torch.Tensor], target: torch.Tensor):
    pos_inds = target == 1
    neg_inds = target < 1

    neg_weights = (1 - target[neg_inds]).pow(_FOCAL_BETA)

    loss = 0
    for pred in preds:
        pred = torch.clamp(pred.sigmoid(), min=1e-4, max=1 - 1e-4)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = (1 - pos_pred).pow(_FOCAL_ALPHA) * torch.log(pos_pred)
        neg_loss = neg_weights * neg_pred.pow(_FOCAL_ALPHA) * torch.log(1 - neg_pred)

        num_objs = pos_pred.nelement()

        if num_objs == 0:
            loss = loss - neg_loss.sum()
        else:
            loss = loss - (pos_loss.sum() + neg_loss.sum()) / num_objs

    return loss / len(preds)


def _embedding_loss(
    tl_embd_list: List[torch.Tensor],
    br_embd_list: List[torch.Tensor],
    valid_mask: torch.Tensor,
):

    num_objs = valid_mask.sum(dim=1, keepdim=True)  # [B, 1]

    pull_loss, push_loss = 0, 0
    for tl_embd, br_embd in zip(tl_embd_list, br_embd_list):
        tl_embd = tl_embd.squeeze()
        br_embd = br_embd.squeeze()

        mean_embd = (tl_embd + br_embd) / 2

        # ---- Pull Loss ----
        dist = (tl_embd - mean_embd) ** 2 + (br_embd - mean_embd) ** 2
        pull = (dist * valid_mask.float()) / (num_objs + 1e-4)  # [B, 1]

        # ---- Push Loss -----
        cross_mask = valid_mask[:, :, None] & valid_mask[:, None, :]
        cross_dist = torch.abs(mean_embd[:, :, None] - mean_embd[:, None, :])

        margin = F.relu(_EMBD_DELTA - cross_dist, inplace=True)
        margin = margin * cross_mask.float()

        # sum across all objs and then subtract the diagonal sum (j == k)
        pair_sum = margin.sum(dim=(1, 2))
        pair_sum = pair_sum[:, None] - _EMBD_DELTA * num_objs

        push = pair_sum / (num_objs * (num_objs - 1) + 1e-4)  # [B, 1]

        push_loss += push.sum()
        pull_loss += pull.sum()

    return pull_loss / len(tl_embd_list), push_loss / len(tl_embd_list)


def _regs_loss(
    regs: List[torch.Tensor], gt_reg: torch.Tensor, valid_mask: torch.Tensor
):

    num_objs = valid_mask.float().sum()
    mask = valid_mask[:, :, None].expand_as(gt_reg)
    loss = sum(
        F.smooth_l1_loss(r[mask], gt_reg[mask], reduction="sum") / num_objs
        for r in regs
    )
    return loss / len(regs)


class Loss(nn.Module):
    def forward(
        self, batch: Dict[str, torch.Tensor], output: List[Dict[str, torch.Tensor]]
    ):
        # Convert each [B, H, W, C] to [B, num_obj, C] where C=1 for embd and C=2 for regs
        # By gathering along spatial (H x W) dim with the flattened indices from gt
        tl_hmap = [x["tl_hmap"] for x in output]
        br_hmap = [x["br_hmap"] for x in output]
        tl_embd = [
            transpose_and_gather_feat(x["tl_embd"], ind=batch["tl_indices"])
            for x in output
        ]
        br_embd = [
            transpose_and_gather_feat(x["br_embd"], ind=batch["br_indices"])
            for x in output
        ]
        tl_regs = [
            transpose_and_gather_feat(x["tl_regs"], ind=batch["tl_indices"])
            for x in output
        ]
        br_regs = [
            transpose_and_gather_feat(x["br_regs"], ind=batch["br_indices"])
            for x in output
        ]

        focal_loss = _neg_loss(tl_hmap, batch["tl_hmap"]) + _neg_loss(
            br_hmap, batch["br_hmap"]
        )
        reg_loss = _regs_loss(
            tl_regs, batch["tl_regs"], valid_mask=batch["ind_masks"]
        ) + _regs_loss(br_regs, batch["br_regs"], valid_mask=batch["ind_masks"])
        pull_loss, push_loss = _embedding_loss(
            tl_embd, br_embd, valid_mask=batch["ind_masks"]
        )

        loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
        return {
            "loss": loss,
            "focal_loss": focal_loss,
            "reg_loss": reg_loss,
            "pull_loss": pull_loss,
            "push_loss": push_loss,
        }
