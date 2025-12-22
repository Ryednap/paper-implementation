from __future__ import annotations
import os
from typing import Optional, cast, override
import lightning as L
import loguru
import numpy as np
import orjson
import torch
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRSchedulerConfig,
)
from wandb import Run
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from nms import soft_nms_merge
from .hourglass import CenterNetHourglass
from configs.config import Config
from loss import centernet_loss
from kp_utils import (
    decode_dets,
    rescale_dets,
    center_match,
    convert_to_coco_eval_format,
)
from debugger import tdebug


class CocoValidatorCallback:
    """
    Callback to perform coco validation routine. At the end of each batch
    validation step it collects the detection results and respective image id
    and then at the end of epoch validation it creates validation routine to
    get current scores.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

        annotation_dir = os.path.join(
            cfg.val_data_dir, "annotations", f"instances_val2014.json"
        )
        self._coco = COCO(annotation_file=annotation_dir)

        self._best_validation = 0.0
        self._results = {}

    def setup(self, fabric: L.Fabric, logger: "loguru.Logger"):
        self.fabric = fabric
        self.logger = logger

    def on_validation_epoch_start(self, epoch: int):
        self.logger.info("Epoch {} Starting validation..", epoch)
        self._results = {}

    def on_validation_batch_end(self, outputs):
        image_id, r = outputs
        self._results[image_id] = r

    def on_validation_epoch_end(self, epoch: int, model: L.LightningModule):
        if not self.fabric.is_global_zero:
            return

        detections = convert_to_coco_eval_format(self._results)
        if self.cfg.val_data_dir is not None:
            result_json = os.path.join(
                self.cfg.val_data_dir, f"epoch{epoch}_results.json"
            )
            with open(result_json, "wb") as f:
                f.write(orjson.dumps(detections))

        coco_dets = self._coco.loadRes(detections)  # type: ignore
        coco_eval = COCOeval(self._coco, coco_dets, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.logger.info("Coco Eval Stats {}", coco_eval.stats)

        current = coco_eval.stats[0]
        self.logger.info(
            "Epoch {} Validation result Current {} Best {}",
            epoch,
            current,
            self._best_validation,
        )
        if current > self._best_validation:
            save_dir = os.path.join(self.cfg.ckpt_dir, "models")
            save_path = os.path.join(save_dir, f"best_{self._best_validation}.ckpt")
            os.makedirs(save_dir, exist_ok=True)
            self.fabric.save(save_path, model.state_dict())
            self.logger.info("Epoch {} Saved Best Model at {}", epoch, save_path)


class CenterNet(L.LightningModule):
    def __init__(
        self, cfg: Config, logger: "loguru.Logger", wandb: Optional[Run] = None
    ):
        super().__init__()

        self.cfg = cfg
        self.wandb = wandb
        self.logging_frequency = self.cfg.logging_frequency
        self.my_logger = logger
        self.model = CenterNetHourglass(
            n=cfg.n,
            num_classes=cfg.num_classes,
            nstack=cfg.nstack,
            dims=cast(list, cfg.dims),
            num_modules=cast(list, cfg.num_modules),
            conv_dim=cfg.conv_dim,
            deep_supervision=cfg.deep_supervision,
        )

        self.loss_fn = centernet_loss

        self._total_steps = 0

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image = batch["image"]
        output_list = self(image)
        loss_dict = self.loss_fn(batch=batch, output=output_list)

        if self._total_steps % self.logging_frequency == 0 and self.wandb is not None:
            self.wandb.log(loss_dict)

        self._total_steps += 1
        return {"output": output_list, "loss_dict": loss_dict}

    def validation_step(self, batch, batch_idx):
        image_id, input_dict = batch
        detections_list, centers_list = [], []
        for scale, d in input_dict.items():
            image = d["image"]
            out_list = self(image)
            out_dict = out_list[-1]
            dets = decode_dets(
                tl_hmap=out_dict["tl_hmap"],
                br_hmap=out_dict["br_hmap"],
                ct_hmap=out_dict["ct_hmap"],
                tl_regs=out_dict["tl_regs"],
                br_regs=out_dict["br_regs"],
                ct_regs=out_dict["ct_regs"],
                tl_embd=out_dict["tl_embd"],
                br_embd=out_dict["br_embd"],
                topk_K=self.cfg.topk_k,
                nms_kernel=3,
                ae_threshold=self.cfg.ae_threshold,
                num_dets=self.cfg.num_dets,
            )
            # self.my_logger.opt(lazy=True).debug("Det debug:\n{dbg}", dbg=lambda: tdebug(dets))
            bboxes, scores, centers = rescale_dets(
                bboxes=dets["bboxes"],
                scores=dets["scores"],
                centers=dets["centers"],
                ratios=d["ratios"],
                borders=d["borders"],
                original_size=d["size"],
            )
            bboxes /= scale
            centers[..., 0:2] /= scale

            detections = torch.cat(
                [bboxes, scores, dets["tl_scores"], dets["br_scores"], dets["classes"]],
                dim=-1,
            )
            detections = detections.reshape(1, -1, 8)
            centers = centers.reshape(1, -1, 4)
            detections_list.append(detections)
            centers_list.append(centers)

        # concatenate along the num_dets axis but note that we are assuming batch=1
        all_detections = torch.concatenate(detections_list, dim=1)[0]
        all_centers = torch.concatenate(centers_list, dim=1)[0]

        all_detections, all_classes = center_match(all_detections, all_centers)
        # # reject detections with negative scores.
        # all_detections = all_detections[all_detections[..., 4] > -1]

        all_detections = all_detections.cpu().numpy()
        all_classes = all_classes.cpu().numpy()

        r = {}
        for j in range(self.cfg.num_classes):
            keep_inds = all_classes == j
            r[j + 1] = all_detections[keep_inds][:, 0:7].astype(np.float32)
            soft_nms_merge(
                r[j + 1],
                Nt=self.cfg.nms_threshold,
                method=2,
                weight_exp=self.cfg.nms_gaussian_w_exp,
            )
            # only bboxes and scores
            r[j + 1] = r[j + 1][:, 0:5]

        scores = np.hstack([r[j][:, -1] for j in range(1, self.cfg.num_classes + 1)])

        # Note since the r is trivially sorted by classes we cannot
        # do simly sort by scores and then just select :max_objs.
        # But I wonder why not maybe time complexity?
        if len(scores) > self.cfg.max_objs:
            kth = len(scores) - self.cfg.max_objs
            thresh = np.partition(scores, kth=kth)[kth]
            for j in range(1, self.cfg.num_classes + 1):
                keep_inds = r[j][:, -1] >= thresh
                r[j] = r[j][keep_inds]

        return image_id, r

    @override
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=self.cfg.lr_step_size,
            gamma=self.cfg.lr_gamma,
        )
        return OptimizerLRSchedulerConfig(
            optimizer=optimizer,
            lr_scheduler=LRSchedulerConfigType(
                scheduler=scheduler,
                interval="step",
                frequency=1,
            ),
        )
