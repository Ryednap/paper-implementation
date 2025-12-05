import torch
import torch.nn as nn
import torch.nn.functional as F

from .keypoint import _transpose_and_gather_feat


# deep supervision where each preds have same shape as targets
def _neg_loss(preds: list[torch.Tensor], targets: torch.Tensor):
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


def _embedding_loss(
    embd0s: list[torch.Tensor], embd1s: list[torch.Tensor], mask: torch.Tensor
):

    num = mask.sum(dim=1, keepdim=True).float()  # (B, 1)

    pull, push = 0, 0
    for embd0, embd1 in zip(embd0s, embd1s):
        embd0 = embd0.squeeze()  # [B, num_obj]
        embd1 = embd1.squeeze()  # [B, num_obj]

        embd_mean = (embd0 + embd1) / 2  # (B, num_obj)

        embd0 = torch.pow(embd0 - embd_mean, 2) / (num + 1e-4)  # (B, num_obj)
        embd0 = embd0[mask].sum()
        embd1 = torch.pow(embd1 - embd_mean, 2) / (num + 1e-4)  # (B, num_obj)
        embd1 = embd1[mask].sum()
        pull += embd0 + embd1

        push_mask = (mask[:, None, :] + mask[:, :, None]) == 2  # (B, num_obj, num_obj)
        dist = F.relu(
            1 - torch.abs(embd_mean[:, None, :] - embd_mean[:, :, None]), inplace=True
        )  # (B, num_obj, num_obj)
        dist = dist / ((num - 1) * num + 1e-4)[:, :, None]
        dist = dist[push_mask].sum()
        push += dist

    return pull / len(embd0s), push / len(embd0s)



def _reg_loss(regs, gt_regs, mask):
    num = mask.float().sum() + 1e-4
    mask = mask[:, :, None].expand_as(gt_regs)  # [B, num_obj, 2]
    loss = sum(
        [F.smooth_l1_loss(r[mask], gt_regs[mask], reduction="sum") / num for r in regs]
    )
    return loss / len(regs)


class Loss(nn.Module):
  def __init__(self, model):
    super(Loss, self).__init__()
    self.model = model

  def forward(self, batch):
    outputs = self.model(batch['image'])
    hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*outputs)

    embd_tl = [_transpose_and_gather_feat(e, batch['inds_tl']) for e in embd_tl]
    embd_br = [_transpose_and_gather_feat(e, batch['inds_br']) for e in embd_br]
    regs_tl = [_transpose_and_gather_feat(r, batch['inds_tl']) for r in regs_tl]
    regs_br = [_transpose_and_gather_feat(r, batch['inds_br']) for r in regs_br]

    focal_loss = _neg_loss(hmap_tl, batch['hmap_tl']) + \
                 _neg_loss(hmap_br, batch['hmap_br'])
    reg_loss = _reg_loss(regs_tl, batch['regs_tl'], batch['ind_masks']) + \
               _reg_loss(regs_br, batch['regs_br'], batch['ind_masks'])
    pull_loss, push_loss = _embedding_loss(embd_tl, embd_br, batch['ind_masks'])

    loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss
    return loss.unsqueeze(0), outputs