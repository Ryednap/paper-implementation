import logging
import os
import sys
import time
import argparse
from datetime import datetime
from Cython import warn
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.distributed as dist
import torch.utils.data.distributed
import nms
import warnings

from datasets.coco import COCO, COCO_eval
from nets.hourglass import get_hourglass
from utils.losses import _neg_loss, _embedding_loss, _reg_loss, Loss
from utils.keypoint import _decode, _rescale_dets, _transpose_and_gather_feat
from utils.utils import count_parameters

warnings.filterwarnings(action="ignore", message="An output with one or more elements was resized")

parser = argparse.ArgumentParser(description="cornernet")

parser.add_argument("--local-rank", type=int, default=0)
parser.add_argument("--dist", action="store_true")

parser.add_argument("--root-dir", type=str, default="./")
parser.add_argument("--data-dir", type=str, default="./data")
parser.add_argument("--log_name", type=str, default="test")
parser.add_argument("--dataset", type=str, default="coco", choices=["coco", "pasacl"])
parser.add_argument("--arch", type=str, default="large_hourglass")
parser.add_argument('--img_size', type=int, default=511)
parser.add_argument("--split_ratio", type=float, default=1.0)
parser.add_argument("--lr", type=float, default=2.5e-4)
parser.add_argument("--lr_step", type=str, default="45,60")
parser.add_argument("--batch-size", type=int, default=48)
parser.add_argument("--num_epochs", type=int, default=70)
parser.add_argument("--log_interval", type=int, default=100)
parser.add_argument("--val_interval", type=int, default=5)
parser.add_argument("--num_workers", type=int, default=2)

cfg = parser.parse_args()

os.chdir(cfg.root_dir)

cfg.log_dir = os.path.join(cfg.root_dir, "logs", cfg.log_name)
cfg.ckpt_dir = os.path.join(cfg.root_dir, "ckpt", cfg.log_name)

os.makedirs(cfg.log_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

cfg.lr_step = [int(s) for s in cfg.lr_step.split(",")]


class Saver:
    def __init__(self, distributed_rank, save_dir):
        self.distributed_rank = distributed_rank
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        return

    def save(self, obj, save_name):
        if self.distributed_rank == 0:
            torch.save(obj, os.path.join(self.save_dir, save_name + ".t7"))
            return "checkpoint saved in %s !" % os.path.join(self.save_dir, save_name)
        else:
            return ""


def create_logger(distributed_rank=0, save_dir=None):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.DEBUG)

    filename = "log_%s.txt" % (datetime.now().strftime("%Y_%m_%d_%H_%M_%S"))

    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    # formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    formatter = logging.Formatter("%(message)s [%(asctime)s]")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir is not None:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def main():
    saver = Saver(cfg.local_rank, save_dir=cfg.ckpt_dir)
    logger = create_logger(cfg.local_rank, save_dir=cfg.log_dir)

    print = logger.info
    if "LOCAL_RANK" in os.environ:
        cfg.local_rank = int(os.environ["LOCAL_RANK"])
    print(cfg)

    torch.manual_seed(317)
    torch.backends.cudnn.benchmark = True  # disable is OOM

    num_gpus = torch.cuda.device_count()
    if cfg.dist:
        cfg.device = torch.device("cuda:%d" % cfg.local_rank)
        torch.cuda.set_device(cfg.local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=num_gpus,
            rank=cfg.local_rank,
        )
    else:
        cfg.device = torch.device("cuda")

    print("Setting up data")

    Dataset = COCO
    train_dataset = Dataset(
        cfg.data_dir, "train", split_ratio=cfg.split_ratio, img_size=cfg.img_size
    )
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        dataset=train_dataset,
        num_replicas=num_gpus,
        rank=cfg.local_rank,
        shuffle=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size // num_gpus if cfg.dist else cfg.batch_size,
        shuffle=not cfg.dist,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
        sampler=train_sampler if cfg.dist else None,
        prefetch_factor=2
    )

    Dataset_eval = COCO_eval
    val_dataset = Dataset_eval(cfg.data_dir, "val", test_scales=[1.0], test_flip=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        collate_fn=val_dataset.collate_fn,
        prefetch_factor=2
    )

    print("Creating model...")
    if "hourglass" in cfg.arch:
        model = get_hourglass[cfg.arch]
        print(count_parameters(model))
    else:
        raise ValueError("Unsupported model")

    model = model.to(cfg.device)
    if cfg.dist:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                cfg.local_rank,
            ],
            output_device=cfg.local_rank,
        )

    optimizer = torch.optim.Adam(model.parameters(), cfg.lr)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.lr_step, gamma=0.1
    )

    def train(epoch):
        print("\n%s Epoch: %d" % (datetime.now(), epoch))
        model.train()

        tic = time.perf_counter()
        for batch_idx, batch in enumerate(train_loader):
            for k in batch:
                batch[k] = batch[k].to(device=cfg.device, non_blocking=True)

            outputs = model(batch["image"])
            hmap_tl, hmap_br, embd_tl, embd_br, regs_tl, regs_br = zip(*outputs)

            embd_tl = [_transpose_and_gather_feat(e, batch["inds_tl"]) for e in embd_tl]
            embd_br = [_transpose_and_gather_feat(e, batch["inds_br"]) for e in embd_br]
            regs_tl = [_transpose_and_gather_feat(r, batch["inds_tl"]) for r in regs_tl]
            regs_br = [_transpose_and_gather_feat(r, batch["inds_br"]) for r in regs_br]

            focal_loss = _neg_loss(hmap_tl, batch["hmap_tl"]) + _neg_loss(
                hmap_br, batch["hmap_br"]
            )

            reg_loss = _reg_loss(
                regs_tl, batch["regs_tl"], batch["ind_masks"]
            ) + _reg_loss(regs_br, batch["regs_br"], batch["ind_masks"])

            pull_loss, push_loss = _embedding_loss(embd_tl, embd_br, batch["ind_masks"])

            loss = focal_loss + 0.1 * pull_loss + 0.1 * push_loss + reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % cfg.log_interval == 0:
                duration = time.perf_counter() - tic
                tic = time.perf_counter()
                print(
                    "[%d/%d-%d/%d] "
                    % (epoch, cfg.num_epochs, batch_idx, len(train_loader))
                    + " focal_loss= %.5f pull_loss= %.5f push_loss= %.5f reg_loss= %.5f"
                    % (
                        focal_loss.item(),
                        pull_loss.item(),
                        push_loss.item(),
                        reg_loss.item(),
                    )
                    + " (%d samples/sec)"
                    % (cfg.batch_size * cfg.log_interval / duration)
                )
        return

    def val_map(epoch):
        print("\n%s Val@Epoch: %d" % (datetime.now(), epoch))
        model.eval()

        results = {}
        with torch.no_grad():
            for inputs in val_loader:
                img_id, inputs = inputs[0]

                detections = []
                for scale in inputs:
                    inputs[scale]["image"] = inputs[scale]["image"].to(cfg.device)
                    output = model(inputs[scale]['image'])[-1]
                    det = _decode(*output, ae_threshold=0.5, K=100, kernel=3)
                    det = det.reshape(det.shape[0], -1, 8).detach().cpu().numpy()
                    if det.shape[0] == 2:
                        det[1, :, [0, 2]] = (
                            inputs[scale]["fmap_size"][0, 1] - det[1, :, [2, 0]]
                        )
                        det = det.reshape(1, -1, 8)

                    _rescale_dets(
                        detections=det,
                        ratios=inputs[scale]["ratio"],
                        borders=inputs[scale]["border"],
                        sizes=inputs[scale]["size"],
                    )

                    det[:, :, 0:4] /= scale
                    detections.append(det)

                detections = np.concatenate(detections, axis=1)[0]
                # reject detections with negative scores
                detections = detections[detections[:, 4] > -1]
                classes = detections[..., -1]

                results[img_id] = {}
                for j in range(val_dataset.num_classes):
                    keep_inds = classes == j
                    results[img_id][j + 1] = detections[keep_inds][:, 0:7].astype(
                        np.float32
                    )
                    nms.soft_nms_merge(
                        results[img_id][j + 1], Nt=0.5, method=2, weight_exp=10
                    )
                    # soft_nms(results[img_id][j + 1], Nt=0.5, method=2)
                    results[img_id][j + 1] = results[img_id][j + 1][:, 0:5]

                scores = np.hstack(
                    [
                        results[img_id][j][:, -1]
                        for j in range(1, val_dataset.num_classes + 1)
                    ]
                )
                if len(scores) > val_dataset.max_objs:
                    kth = len(scores) - val_dataset.max_objs
                    thresh = np.partition(scores, kth)[kth]
                    for j in range(1, val_dataset.num_classes + 1):
                        keep_inds = results[img_id][j][:, -1] >= thresh
                        results[img_id][j] = results[img_id][j][keep_inds]

        eval_results = val_dataset.run_eval(results, save_dir=cfg.ckpt_dir)
        print(eval_results)
            
    print("Starting training...")
    for epoch in range(cfg.num_epochs + 1):
        train_sampler.set_epoch(epoch)
        train(epoch)
        if cfg.val_interval > 0 and epoch % cfg.val_interval == 0:
            val_map(epoch)
        
        print(saver.save(model.state_dict(), "checkpoint"))
        lr_scheduler.step(epoch)


class DisablePrint:
  def __init__(self, local_rank=0):
    self.local_rank = local_rank

  def __enter__(self):
    if self.local_rank != 0:
      self._original_stdout = sys.stdout
      sys.stdout = open(os.devnull, 'w')
    else:
      pass

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.local_rank != 0:
      sys.stdout.close()
      sys.stdout = self._original_stdout
    else:
      pass


if __name__ == '__main__':
    with DisablePrint(local_rank=cfg.local_rank):
        main()