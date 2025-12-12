import logging
from matplotlib.transforms import BboxBase
import numpy as np
import orjson
import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from typing import Any, Dict, Optional
from lightning.fabric import Fabric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from configs.base import Config
from utils import set_seed, topk, transpose_and_gather_feat, hetmap_nms
from nms import soft_nms_merge
from datasets._coco_constants import COCO_IDS


def gather_feat(feat: torch.Tensor, ind: torch.Tensor):
    """
    Returns the value of the feature tensor at given indices
    Expected Tensor shape::
    `feat`: (Batch, K, C)
    `ind`: (Batch, num_dets)
    The gathering is performed along dim=1
    """

    B, num_dets = ind.shape
    ind = ind[:, :, None].expand(B, num_dets, feat.shape[-1])
    return feat.gather(dim=1, index=ind)


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


class Validator:
    def __init__(self, cfg: Config, logger: logging.Logger):
        self.cfg = cfg
        self.logger = logger
        annotation_dir = cfg.data_dir / "annotations" / f"instances_val2014.json"
        self._coco = COCO(annotation_file=annotation_dir)

    @staticmethod
    def decode(
        tl_hmap: torch.Tensor,
        br_hmap: torch.Tensor,
        tl_embd: torch.Tensor,
        br_embd: torch.Tensor,
        tl_regs: torch.Tensor,
        br_regs: torch.Tensor,
        topk_k: int,
        nms_kernel: int,
        ae_threshold: float,
        num_dets: int = 1000,
    ):
        B, C, H, W = tl_hmap.shape

        tl_hmap = torch.sigmoid(tl_hmap)
        br_hmap = torch.sigmoid(br_hmap)

        # perform nms on heatmap
        tl_hmap = hetmap_nms(heatmap=tl_hmap, kernel=nms_kernel)
        br_hmap = hetmap_nms(heatmap=br_hmap, kernel=nms_kernel)

        tl_scores, tl_inds, tl_classes, tl_ys, tl_xs = topk(tl_hmap, k=topk_k)
        br_scores, br_inds, br_classes, br_ys, br_xs = topk(tl_hmap, k=topk_k)

        tl_xs = tl_xs.view(B, topk_k, 1).expand(B, topk_k, topk_k)
        br_xs = br_xs.view(B, 1, topk_k).expand(B, topk_k, topk_k)
        tl_ys = tl_ys.view(B, topk_k, 1).expand(B, topk_k, topk_k)
        br_ys = br_ys.view(B, 1, topk_k).expand(B, topk_k, topk_k)

        tl_regs = transpose_and_gather_feat(tl_regs, ind=tl_inds)
        br_regs = transpose_and_gather_feat(br_regs, ind=br_inds)
        tl_embd = transpose_and_gather_feat(tl_embd, ind=tl_inds)
        br_embd = transpose_and_gather_feat(br_embd, ind=br_embd)

        tl_regs.view(B, topk_k, 1, 2)
        br_regs.view(B, 1, topk_k, 2)

        tl_xs = tl_xs + tl_regs[..., 0]
        tl_ys = tl_ys + tl_regs[..., 1]
        br_xs = br_xs + br_regs[..., 0]
        br_ys = br_ys + br_regs[..., 1]

        # all possible boxes based on top k corners (ignoring classes)
        bboxes = torch.stack((tl_xs, tl_ys, br_xs, br_ys), dim=3)  # [B, K, K, 4]

        tl_embd = tl_embd.view(B, topk_k, 1)
        br_embd = br_embd.view(B, 1, topk_k)

        dists = torch.abs(tl_embd - br_embd)

        tl_scores = tl_scores.view(B, topk_k, 1).expand(B, topk_k, topk_k)
        br_scores = br_scores.view(B, 1, topk_k).expand(B, topk_k, topk_k)

        score_matrix = (tl_scores + br_scores) / 2

        # reject boxes based on classes where tl and br don't agree
        tl_classes = tl_classes.view(B, topk_k, 1).expand(B, topk_k, topk_k)
        br_classes = br_classes.view(B, 1, topk_k).expand(B, topk_k, topk_k)
        cls_inds = tl_classes != br_classes

        # reject boxes based on distances
        dist_inds = dists > ae_threshold

        # reject boxes based on invalid boxes
        width_inds = br_xs < tl_xs
        height_inds = br_ys < tl_ys

        score_matrix[cls_inds] = -1
        score_matrix[dist_inds] = -1
        score_matrix[width_inds] = -1
        score_matrix[height_inds] = -1

        # flatten the score matrix
        scores = score_matrix.view(B, -1)
        scores, inds = torch.topk(scores, k=num_dets)
        scores = scores[:, :, None]  # [B, num_dets, 1]

        bboxes = bboxes.view(B, -1, 4)  # [B, K * K, 4]
        bboxes = gather_feat(bboxes, inds)  # [B, num_dets, 4]

        classes = tl_classes.contiguous().view(B, -1, 1)  # [B, K * K, 1]
        classes = gather_feat(classes, inds)  # [B, num_dets, 1]

        tl_scores = tl_scores.contiguous().view(B, -1, 1)  # [B, K * K, 1]
        br_scores = br_scores.contiguous().view(B, -1, 1)  # [B, K * K, 1]
        tl_scores = gather_feat(tl_scores, inds)
        br_scores = gather_feat(br_scores, inds)

        return {
            "bboxes": bboxes,
            "classes": classes,
            "scores": scores,
            "tl_scores": tl_scores,
            "br_scores": br_scores,
        }

    @staticmethod
    def rescale_dets(
        bboxes: torch.Tensor,
        ratios: torch.Tensor,
        borders: torch.Tensor,
        original_sizes: torch.Tensor,
    ):
        xs, ys = bboxes[..., 0:4:2], bboxes[..., 1:4:2]
        xs /= ratios[:, 1][:, None, None]
        ys /= ratios[:, 0][:, None, None]

        # subract top-left coordinate of border
        xs -= borders[:, 2][:, None, None]
        ys -= borders[:, 0][:, None, None]

        xs = xs.clamp(min=0, max=original_sizes[:, 1][:, None, None])
        ys = ys.clamp(min=0, max=original_sizes[:, 0][:, None, None])

        out = bboxes.clone()
        out[..., 0:4:2] = xs
        out[..., 1:4:2] = ys

        return out

    def validate(self, epoch: int, model: nn.Module, val_loader: DataLoader):
        results = {}
        for image_id, input_dict in val_loader:
            detections = []
            for scale, d in input_dict.items():
                # Note that image can be batches
                # of test time augmentations.
                image = d["image"]
                out_dict = model({"image": image})
                det = self.decode(
                    out_dict["tl_hmap"],
                    out_dict["br_hmap"],
                    out_dict["tl_embd"],
                    out_dict["br_embd"],
                    out_dict["tl_regs"],
                    out_dict["br_regs"],
                    topk_k=self.cfg.topk_k,
                    nms_kernel=3,
                    ae_threshold=self.cfg.ae_threshold,
                    num_dets=self.cfg.num_dets,
                )
                bboxes = self.rescale_dets(
                    det["bboxes"],
                    ratios=d["ratio"],
                    borders=d["border"],
                    original_sizes=d["original_size"],
                )
                bboxes /= scale

                scores = det["scores"]
                classes = det["classes"]

                # filter by negative score
                score_filter = scores > -1
                bboxes = bboxes[score_filter].view(1, -1, 4)[0]
                classes = classes[score_filter].view(1, -1, 1)[0]
                scores = scores[score_filter].view(1, -1, 1)[0]

                # [num_dets, 6] with first 4 being bboxes and then scores, classes
                detections.append(torch.stack([bboxes, scores, classes], dim=-1))

            combined_det = torch.concatenate(detections, dim=0).cpu().numpy()
            combined_classes = combined_det[..., 5]

            r = {}
            for j in range(self.cfg.num_classes):
                keep_inds = combined_classes == j
                r[j + 1] = combined_det[keep_inds].astype(np.float32)
                soft_nms_merge(
                    r[j + 1],
                    Nt=self.cfg.nms_threshold,
                    method=2,
                    weight_exp=self.cfg.nms_gaussian_w_exp,
                )

            scores = np.hstack(
                [r[j][:, -1] for j in range(1, self.cfg.num_classes + 1)]
            )

            # Note since the r is trivially sorted by classes we cannot
            # do simly sort by scores and then just select :max_objs.
            # But I wonder why not maybe time complexity?
            if len(scores) > self.cfg.max_objs:
                kth = len(scores) - self.cfg.max_objs
                thresh = np.partition(scores, kth=kth)[kth]
                for j in range(1, self.cfg.num_classes + 1):
                    keep_inds = r[j + 1][:, -1] >= thresh
                    r[j + 1] = r[j][keep_inds]

            results[image_id] = r

        detections = convert_to_coco_eval_format(results)
        if self.cfg.eval_save_dir is not None:
            result_json = self.cfg.eval_save_dir / f"epoch{epoch}_results.json"
            with open(result_json, "wb") as f:
                f.write(orjson.dumps(detections))

        coco_dets = self._coco.loadRes(detections)
        coco_eval = COCOeval(self._coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        return coco_eval.stats


class Trainer:
    def __init__(
        self,
        fabric: Fabric,
        cfg: Config,
        logger: logging.Logger,
        validation_frequency: int = 1,
        logging_frequency: int = 1,
        disable_tqdm: bool = True,
    ):
        self.fabric = fabric
        self.cfg = cfg
        self.logger = logger
        self.validation_frequency = validation_frequency
        self.logging_frequency = logging_frequency
        self.disable_tqdm = disable_tqdm
        self.validator = Validator(self.cfg, logger=self.logger)

        self._last_validation = 0.0
        self._best_validation = 0.0
        self._total_steps = 0

    def fit(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: _LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):

        for epoch in range(self.cfg.num_epochs):
            set_seed(self.cfg.seed + epoch + self.fabric.local_rank)

            model.train()
            self.train_loop(epoch, model, optimizer, train_loader)
            scheduler.step()
            if epoch % self.validation_frequency == 0:
                self._last_validation = self.validator.validate(
                    epoch, model, val_loader
                )
                if self._last_validation > self._best_validation:
                    self._best_validation = self._last_validation
                    fname = f"epoch_{epoch}_metric_{self._best_validation:%.5f}.pth"
                    self.fabric.save(fname, state=model.state_dict())
                    self.logger.info(
                        f"Epoch: %d New Best :: %.5f", epoch, self._best_validation
                    )
                else:
                    self.logger.info(
                        f"Epoch:%d, Curr:%.5f, Best:%5f",
                        epoch,
                        self._last_validation,
                        self._best_validation,
                    )

    def train_loop(
        self,
        epoch: int,
        model: nn.Module,
        optimizer: Optimizer,
        train_loader: DataLoader,
    ):
        disable_tqdm = self.disable_tqdm or not self.fabric.is_global_zero
        pbar = tqdm(train_loader, disable=disable_tqdm)

        for batch in pbar:
            pbar.update(1)

            out_dict = model(batch)
            loss_dict = out_dict["loss"]
            loss = loss_dict["loss"]

            optimizer.zero_grad()
            self.fabric.backward(loss)
            optimizer.step()

            pbar.set_postfix(
                {
                    "epoch": epoch,
                    "loss": loss.item(),
                    "focal_loss": loss_dict["focal_loss"].item(),
                    "push_loss": loss_dict["push_loss"].item(),
                    "pull_loss": loss_dict["pull_loss"].item(),
                    "reg_loss": loss_dict["reg_loss"].item(),
                    "lr": optimizer.param_groups[0]["lr"],
                    "last_validation": self._last_validation,
                    "best_validation": self._best_validation,
                }
            )

            self._total_steps += 1
            if self.logging_frequency % self._total_steps == 0:
                self.logger.info(
                    "Epoch %d/%d, Iteration %d, Loss: %.4f, LR: %.6f",
                    epoch,
                    self.cfg.num_epochs,
                    self._total_steps,
                    loss.item(),
                    optimizer.param_groups[0]["lr"],
                )
