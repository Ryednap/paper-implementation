from pydantic import BaseModel
from typing import Literal, Sequence, Tuple

_PRECISION = Literal["32-true", "16-mixed", "bf16-mixed", "16-true", "bf16-true"]


class Config(BaseModel, extra="allow"):
    # dirs and seed
    seed: int
    train_data_dir: str
    val_data_dir: str
    eval_save_dir: str
    ckpt_dir: str

    # data-prep settings
    use_gaussian: bool
    gaussian_iou: float
    down_ratio: int
    max_objs: int
    num_classes: int
    num_dets: int
    test_scales: Sequence[float]
    topk_k: int
    nms_threshold: float
    nms_gaussian_w_exp: float
    ae_threshold: float
    val_cache_rate: float

    # training setttings
    precision: _PRECISION
    train_patch_size: Tuple[int, int]
    batch_size: int
    max_steps: int
    lr: float
    lr_step_size: int
    lr_gamma: float
    val_frequency: int
    logging_frequency: int
    ckpt_frequency: int


    # Model settings
    n: int
    nstack: int
    conv_dim: int
    dims: Sequence[int]
    num_modules: Sequence[int]
    deep_supervision: bool


OVERRIDABLES = [
    "train_data_dir",
    "val_data_dir",
    "eval_save_dir",
    "ckpt_dir",
    "batch_size",
    "lr",
    "max_steps",
    "precision",
    "val_cache_rate",
]
