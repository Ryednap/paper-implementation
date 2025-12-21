import os
import sys
import tomllib
import loguru
from typing import Optional
import click
import lightning as L
from torch.utils.data import DataLoader


from utils import set_seed
from configs import add_click_overides, Config, OVERRIDABLES
from datasets.coco import CocoTrainDataset, CocoValDataset
from nets.net import CenterNet, CocoValidatorCallback
from trainer import Trainer, INT_MAX


def _get_logger(is_rank_zero: bool, log_path: Optional[str]):
    loguru.logger.remove()

    if not is_rank_zero:
        return loguru.logger

    fmt = "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message}"
    loguru.logger.add(sys.stderr, level="DEBUG", format=fmt)

    if log_path:
        p = os.path.join(log_path, "train_{time:YYYY-MM-DD_HH-mm-ss}.log")
        loguru.logger.add(p, level="DEBUG", format=fmt, enqueue=True)

    return loguru.logger


def train(
    fabric: L.Fabric,
    cfg: Config,
    disable_tqdm: bool,
    limit_train_batches: int,
    limit_val_batches: int,
    num_workers: int,
    log_path: Optional[str],
):
    set_seed(cfg.seed)
    train_dset = CocoTrainDataset(cfg=cfg)
    val_dset = CocoValDataset(cfg=cfg, device=fabric.device)

    train_loader = DataLoader(
        train_dset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_loader = DataLoader(val_dset, batch_size=1, shuffle=False)
    logger = _get_logger(fabric.is_global_zero, log_path)
    model = CenterNet(cfg=cfg, logger=logger)
    trainer = Trainer(
        fabric=fabric,
        logger=logger,
        ckpt_dir=cfg.ckpt_dir,
        max_epochs=cfg.max_epochs,
        validation_frequency=cfg.val_frequency,
        logging_frequency=cfg.logging_frequency,
        ckpt_frequency=cfg.ckpt_frequency,
        disable_tqdm=disable_tqdm,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_val_batches,
    )
    fabric.call("setup", fabric, logger)

    trainer.fit(model, train_loader, val_loader)


@click.command()
@click.option("--config-path", type=click.Path(exists=True), required=False)
@click.option("--log-path", type=click.Path(exists=True), required=False)
@click.option("--num-workers", type=int, required=False, default=1)
@click.option("--disable-tqdm", is_flag=True)
@click.option("--limit-train-batches", type=int, required=False, default=INT_MAX)
@click.option("--limit-val-batches", type=int, required=False, default=INT_MAX)
@add_click_overides(Config, OVERRIDABLES)
def main(
    config_path: Optional[str],
    log_path: Optional[str],
    num_workers: int,
    disable_tqdm: bool,
    limit_train_batches: int,
    limit_val_batches: int,
    **kwargs
):
    overrides = {k: v for k, v in kwargs.items() if v is not None}

    toml_cfg = {}
    if config_path is not None:
        with open(config_path, "rb") as f:
            toml_cfg = tomllib.load(f)

    merged = {**toml_cfg, **overrides}
    cfg = Config.model_validate(merged)

    fabric = L.Fabric(
        accelerator="cuda",
        strategy="auto",
        devices=1,
        precision=cfg.precision,
        callbacks=CocoValidatorCallback(cfg=cfg),
    )

    fabric.launch(
        train,
        cfg,
        disable_tqdm,
        limit_train_batches,
        limit_val_batches,
        num_workers,
        log_path,
    )


if __name__ == "__main__":
    main()
