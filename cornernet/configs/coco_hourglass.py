from dataclasses import dataclass

from configs.base import BaseConfig, Config


@dataclass(slots=True, kw_only=True)
class CocoHourglassSmall(BaseConfig):

    seed = 317

    precision = "32-true"
    train_patch_size = (511, 511)
    train_cache_rate = 0.0
    val_cach_rate = 0.0
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


class CocoHourglassLarge(CocoHourglassSmall):

    nstack = 2
