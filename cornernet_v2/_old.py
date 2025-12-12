from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import transpose_and_gather_feat


def _neg_loss(preds, targets):
    pos_inds = targets == 1  # todo targets > 1-epsilon ?
    neg_inds = targets < 1  # todo targets < 1-epsilon ?

    neg_weights = torch.pow(1 - targets[neg_inds], 4)

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_pred = pred[pos_inds]
        neg_pred = pred[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if pos_pred.nelement() == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
    return loss / len(preds)


def _ae_loss(embd0s, embd1s, mask):
    num = mask.sum(dim=1, keepdim=True).float()  # [B, 1]

    pull, push = 0, 0
    for embd0, embd1 in zip(embd0s, embd1s):
        embd0 = embd0.squeeze()  # [B, num_obj]
        embd1 = embd1.squeeze()  # [B, num_obj]

        embd_mean = (embd0 + embd1) / 2

        embd0 = torch.pow(embd0 - embd_mean, 2) / (num + 1e-4)
        embd0 = embd0[mask].sum()
        embd1 = torch.pow(embd1 - embd_mean, 2) / (num + 1e-4)
        embd1 = embd1[mask].sum()
        pull += embd0 + embd1

        push_mask = (mask[:, None, :] + mask[:, :, None]) == 2  # [B, num_obj, num_obj]
        dist = F.relu(
            1 - (embd_mean[:, None, :] - embd_mean[:, :, None]).abs(), inplace=True
        )
        dist = dist - 1 / (num[:, :, None] + 1e-4)  # substract diagonal elements
        dist = dist / ((num - 1) * num + 1e-4)[:, :, None]  # total num element is n*n-n
        push += dist[push_mask].sum()
    return pull / len(embd0s), push / len(embd0s)


def _reg_loss(regs, gt_regs, mask):
    num = mask.float().sum() + 1e-4
    mask = mask[:, :, None].expand_as(gt_regs)  # [B, num_obj, 2]
    loss = sum(
        [F.smooth_l1_loss(r[mask], gt_regs[mask], reduction="sum") / num for r in regs]
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
        reg_loss = _reg_loss(
            tl_regs, batch["tl_regs"], mask=batch["ind_masks"]
        ) + _reg_loss(br_regs, batch["br_regs"], mask=batch["ind_masks"])
        pull_loss, push_loss = _ae_loss(
            tl_embd, br_embd, mask=batch["ind_masks"]
        )

        loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
        return {
            "loss": loss,
            "focal_loss": focal_loss,
            "reg_loss": reg_loss,
            "pull_loss": pull_loss,
            "push_loss": push_loss,
        }
