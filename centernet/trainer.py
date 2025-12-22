import os
import loguru
import torch
import loguru
import lightning as L
from tqdm import tqdm
from typing import Any, Dict, Iterable, Literal, Optional, cast

from utils import count_parameters, human_format

INT_MAX = 2147483647


class Trainer:
    def __init__(
        self,
        fabric: L.Fabric,
        logger: "loguru.Logger",
        ckpt_dir: Optional[str] = None,
        max_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        limit_train_batches: int = INT_MAX,
        limit_val_batches: int = INT_MAX,
        validation_frequency: int = 1,
        logging_frequency: int = 1,
        ckpt_frequency: int = 1,
        disable_tqdm: bool = False,
    ):

        if not max_steps and not max_epochs:
            raise ValueError("One of `max_steps` or `max_epochs` needs to be supplied.")

        self.fabric = fabric
        self.ckpt = ckpt_dir
        self.logger = logger
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches

        self.ckpt_dir = ckpt_dir
        self.ckpt_frequency = ckpt_frequency
        self.logging_frequency = logging_frequency
        self.validation_frequency = validation_frequency
        self.disable_tqdm = disable_tqdm

        self._current_epoch = 0
        self._current_steps = 0

    def fit(
        self,
        model: L.LightningModule,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        self.logger.info(
            "Total Model Parameters: {}", human_format(count_parameters(model))
        )

        train_loader, val_loader = self.fabric.setup_dataloaders(
            train_loader, val_loader, use_distributed_sampler=False
        )
        optimizer_cfg = model.configure_optimizers()
        assert isinstance(optimizer_cfg, dict)
        assert "optimizer" in optimizer_cfg
        assert "lr_scheduler" in optimizer_cfg
        optimizer, scheduler_cfg = (
            optimizer_cfg["optimizer"],
            optimizer_cfg["lr_scheduler"],
        )
        model, optimizer = self.fabric.setup(model, optimizer)

        state = {
            "model": model,
            "optimizer": optimizer,
            "scheduler": scheduler_cfg,
            "current_epoch": self._current_epoch,
            "current_steps": self._current_steps,
        }

        if self.ckpt_dir is not None:
            ckpt_path = self._get_latest_ckpt(self.ckpt_dir)
            if ckpt_path is not None:
                self.logger.info("Checkpoint {x} found", x=ckpt_path)
                remainder = self.fabric.load(ckpt_path, state)
                self._current_epoch = state["current_epoch"]
                self._current_steps = state["current_steps"]

                # just a healthy reminder.
                if remainder:
                    raise RuntimeError(
                        f"Unused checkpoint keys: {list(remainder.keys())}"
                    )

        while not self.should_stop:
            self.train_loop(
                model=model,
                optimizer=optimizer,
                limit_batches=self.limit_train_batches,
                train_loader=train_loader,
                scheduler_cfg=scheduler_cfg,
            )

            if self._current_epoch % self.validation_frequency == 0:
                self.val_loop(
                    model=model,
                    val_loader=val_loader,
                    limit_batches=self.limit_val_batches,
                )

            self._current_epoch += 1

            if self._current_epoch % self.ckpt_frequency == 0:
                self._create_checkpoint(state)

    def train_loop(
        self,
        model: L.LightningModule,
        optimizer: torch.optim.Optimizer,
        limit_batches: int,
        scheduler_cfg: Dict[str, Any],
        train_loader: torch.utils.data.DataLoader,
    ):
        if limit_batches == 0:
            return
        model.train()
        iterable = self.probar_wrapper(
            train_loader,
            total=min(len(train_loader), limit_batches),
            desc=f"Epoch: {self._current_epoch}",
        )

        for batch_idx, batch in enumerate(iterable):
            if self.should_stop or batch_idx >= limit_batches:
                break

            out_dict = cast(
                dict[str, torch.Tensor], model.training_step(batch, batch_idx)
            )
            loss_dict = cast(dict[str, torch.Tensor], out_dict["loss_dict"])
            loss = loss_dict["loss"]
            optimizer.zero_grad()
            self.fabric.backward(loss)
            optimizer.step()

            self.step_scheduler(
                scheduler_cfg=scheduler_cfg,
                level="step",
                current_value=self._current_steps,
            )

            if self._current_steps % self.logging_frequency == 0:
                loss_str = " ".join(
                    f"{k}={v.item() if isinstance(v, torch.Tensor) else v :.5f}"
                    for k, v in loss_dict.items()
                )

                if self.max_epochs:
                    self.logger.info(
                        "Epoch {}/{} Iteration {} | {}",
                        self._current_epoch,
                        self.max_epochs,
                        self._current_steps,
                        loss_str,
                    )
                else:
                    self.logger.info(
                        "Iteration {}/{} | {}",
                        self._current_steps,
                        self.max_steps,
                        self._current_steps,
                        loss_str,
                    )

            self._current_steps += 1

        self.step_scheduler(
            scheduler_cfg=scheduler_cfg,
            level="epoch",
            current_value=self._current_epoch,
        )

    @torch.inference_mode()
    def val_loop(
        self,
        model: L.LightningModule,
        limit_batches: int,
        val_loader: torch.utils.data.DataLoader,
    ):
        model.eval()
        self.fabric.call("on_validation_epoch_start", epoch=self._current_epoch)

        iterable = self.probar_wrapper(
            val_loader,
            total=min(len(val_loader), limit_batches),
            desc=f"Epoch {self._current_epoch} Validation",
        )

        for batch_idx, batch in enumerate(iterable):
            if batch_idx >= limit_batches:
                break

            self.fabric.call(
                "on_validation_batch_start", batch=batch, batch_idx=batch_idx
            )
            out = model.validation_step(batch, batch_idx)
            self.fabric.call(
                "on_validation_batch_end", out, batch=batch, batch_idx=batch_idx
            )

        self.fabric.call(
            "on_validation_epoch_end", epoch=self._current_epoch, model=model
        )

    def probar_wrapper(self, iterable: Iterable, total: int, **kwargs):
        if self.fabric.is_global_zero and not self.disable_tqdm:
            return tqdm(iterable=iterable, total=total, **kwargs)
        return iterable

    @staticmethod
    def step_scheduler(
        scheduler_cfg: dict[str, Any],
        level: Literal["step", "epoch"],
        current_value: int,
    ):

        if scheduler_cfg["interval"] != level:
            return

        if current_value % scheduler_cfg["frequency"] != 0:
            return

        scheduler_cfg["scheduler"].step()

    @property
    def should_stop(self):
        if self.max_epochs is not None:
            return self._current_epoch >= self.max_epochs
        if self.max_steps is not None:
            return self._current_steps >= self.max_steps

    @staticmethod
    def _get_latest_ckpt(path: str):
        if not path or not os.path.isdir(path):
            return None
        files = sorted([f for f in os.listdir(path) if f.endswith(".ckpt")])
        return os.path.join(path, files[-1]) if files else None

    def _create_checkpoint(self, state: Optional[dict]):
        if state is None or self.ckpt_dir is None:
            return

        state.update(
            current_epoch=self._current_epoch, current_steps=self._current_steps
        )
        ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.ckpt")
        self.fabric.save(ckpt_path, state)
        self.logger.debug(
            "Created Checkpoint {} at epoch {}", ckpt_path, self._current_epoch
        )
