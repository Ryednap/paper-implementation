import os
import click
from dataclasses import asdict
from typing import Any, Dict, Literal, Optional, cast
from lightning import Fabric
from pathlib import Path
from lightning.fabric import Fabric

import torch.optim as optim
from torch.utils.data import DataLoader

from configs.base import Config
from trainer import Trainer
from nets.hourglass import CornerNet
from datasets.coco import CocoTrainDataset, CocoValDataset
from logger import init_logger
from utils import set_seed, count_parameters


def initialize_os_environ():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["TORCH_DISTRIBUTED_BACKEND"] = "nccl"


def train(fabric: Fabric, cfg: Config, disable_tqdm: bool):
    set_seed(cfg.seed)
    initialize_os_environ()

    logger = init_logger(name=__name__, is_rank_zero=fabric.is_global_zero)
    logger.info("Config:\n\n%s\n\n", cfg.to_dict())
    
    train_dset = CocoTrainDataset(cfg=cfg, device=fabric.device)
    val_dset = CocoValDataset(cfg=cfg, device=fabric.device)

    train_loader = DataLoader(
        dataset=train_dset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=8,
    )
    val_loader = DataLoader(
        dataset=val_dset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    train_loader, val_loader = fabric.setup_dataloaders(
        train_loader, val_loader, use_distributed_sampler=True
    )



    model = CornerNet(
        n=cfg.n,
        nstack=cfg.nstack,
        dims=cast(list, cfg.dims),
        num_modules=cast(list, cfg.num_modules),
        num_classes=cfg.num_classes,
    )
    optimizer = optim.Adam(params=model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, cfg.lr_step, gamma=0.1)

    model, optimizer = fabric.setup(model, optimizer)

    logger.info("Total parmeters: %d", count_parameters(model))

    trainer = Trainer(
        fabric=fabric,
        cfg=cfg,
        logger=logger,
        validation_frequency=cfg.val_frequency,
        logging_frequency=cfg.logging_frequency,
        disable_tqdm=disable_tqdm,
    )

    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler=cast(optim.lr_scheduler._LRScheduler, scheduler),
        train_loader=train_loader,
        val_loader=val_loader,
    )


@click.command()
@click.option("--disable-tqdm", is_flag=True)
@click.option(
    "--config",
    type=click.Choice(["coco-hg-small", "coco-hg-large"]),
    required=True,
    help="Config type",
)
@click.option(
    "--precision",
    type=click.Choice(["32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"]),
    required=True,
)
@click.option("--data-dir", type=click.Path(exists=True), required=True)
@click.option("--val-data-dir", type=click.Path(exists=True), required=False)
@click.option("--eval-save-dir", type=click.Path(exists=True), required=True)
@click.option("--seed", type=int, required=False)
@click.option("--train-cache-rate", type=float, required=False)
@click.option("--batch-size", type=int, required=False)
@click.option("--lr", type=float, required=False)
def main(
    disable_tqdm: bool,
    config: str,
    precision: str,
    data_dir: Path,
    val_data_dir: Optional[Path],
    eval_save_dir: Path,
    seed: Optional[int],
    train_cache_rate: Optional[float],
    batch_size: Optional[int],
    lr: Optional[float],
):

    if val_data_dir is None:
        val_data_dir = data_dir

    if config == "coco-hg-small":
        from configs.coco_hourglass import CocoHourglassSmall

        cfg = CocoHourglassSmall()
    elif config == "coco-hg-large":
        from configs.coco_hourglass import CocoHourglassLarge

        cfg = CocoHourglassLarge()

    else:
        raise ValueError("Unrecognized args")

    cfg.data_dir = Path(data_dir)
    cfg.val_data_dir = Path(val_data_dir)
    cfg.eval_save_dir = Path(eval_save_dir)
    cfg.precision = precision

    if seed is not None:
        cfg.seed = seed
    if train_cache_rate is not None:
        cfg.train_cache_rate = train_cache_rate
    if batch_size is not None:
        cfg.batch_size = batch_size
    if lr is not None:
        cfg.lr = lr

    fabric = Fabric(
        accelerator="cuda",
        strategy="auto",
        devices=1,
        precision=cfg.precision,
    )

    fabric.launch(train, cfg, disable_tqdm)  # type: ignore


if __name__ == "__main__":
    main()
