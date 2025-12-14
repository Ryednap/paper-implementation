from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence, Tuple


class Config:
    seed: int
    data_dir: Path
    val_data_dir: Path
    eval_save_dir: Path
    ckpt_dir: Path

    precision: Literal["32-true", "16-mixed", "16-true", "bf16-mixed", "bf16-true"]
    train_patch_size: Tuple[int, int]
    train_cache_rate: float
    val_cache_rate: float
    use_gaussian: bool
    gaussian_iou: float
    batch_size: int
    lr: float
    num_epochs: int
    lr_step: Tuple[int, int]
    val_frequency: int
    logging_frequency: int
    ckpt_frequency: int

    down_ratio: int
    max_objs: int
    num_classes: int
    test_scales: Sequence[float]
    num_dets: int
    topk_k: int
    nms_threshold: float
    ae_threshold: float
    nms_gaussian_w_exp: float

    # Model
    n: int
    nstack: int
    dims: Sequence[int]
    num_modules: Sequence[int]


@dataclass(slots=True, kw_only=True)
class BaseConfig(Config):

    seed = 317
    data_dir = "./data"
    val_data_dir = None
    eval_save_dir = "./eval"

    precision = "32-true"
    train_patch_size = (511, 511)
    train_cache_rate = 0.0
    val_cache_rate = 0.0
    use_gaussian = True
    gaussian_iou = 0.3
    batch_size = 48
    lr = 2.5e-4
    num_epochs = 70
    lr_step = (45, 60)
    val_frequency = 5
    logging_frequency = 100

    down_ratio = 4
    max_objs = 128
    num_classes = 80
    test_scales = [1.0]
    num_dets = 1000
    topk_k = 100
    nms_threshold = 0.5
    ae_threshold = 0.5
    nms_gaussian_w_exp = 10

    n = 5
    nstack = 1
    dims = [256, 256, 384, 384, 384, 512]
    num_modules = [2, 2, 2, 2, 2, 4]

    def __post_init__(self):
        if self.val_data_dir is None:
            self.val_data_dir = self.data_dir

    def to_dict(self):
        """
        Returns a dictionary of all config attributes (including inherited ones),
        with their current values for this instance.
        """
        # Collect all attributes from the class and its parents
        result = {}
        for cls in self.__class__.__mro__:
            if cls is object:
                continue
            for key, value in cls.__dict__.items():
                if not key.startswith("_") and not callable(value):
                    result[key] = getattr(self, key, value)
        # Also include any instance attributes (in case of overrides)
        result.update(self.__dict__)
        return result
