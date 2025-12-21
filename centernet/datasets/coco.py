import math
import os
from typing import Literal
import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pycocotools import coco
from tqdm import tqdm
from joblib import Parallel, delayed

from configs.config import Config
from ._coco_constants import (
    COCO_IDS,
    COCO_EIGEN_VALUES,
    COCO_EIGEN_VECTORS,
    COCO_MEAN,
    COCO_STD,
)
from ._utils import compute_gaussian_radius, draw_gaussian, center_crop_gpu


def _get_data_list(data_dir: str, split: Literal["train", "val", "test"]):
    image_dir = os.path.join(data_dir, "images", f"{split}2014")
    annot_path = os.path.join(data_dir, "annotations", f"instances_{split}2014.json")

    coco_gt = coco.COCO(annot_path)
    image_ids = coco_gt.getImgIds()

    data_list = []
    for img_id in image_ids:
        image_path = os.path.join(image_dir, coco_gt.loadImgs(img_id)[0]["file_name"])

        annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))

        labels = [anno["category_id"] for anno in annotations]
        labels = np.array([COCO_IDS.index(label) for label in labels])
        bboxes = np.array([anno["bbox"] for anno in annotations])

        data_list.append(
            {
                "image_id": img_id,
                "image": image_path,
                "bboxes": bboxes,
                "labels": labels,
            }
        )

    return data_list


def _get_train_transform(train_patch_size: tuple[int, int]):

    def lightning_(image, alpha_std: float = 0.1, rng=np.random, **kwargs):
        alpha = rng.normal(scale=alpha_std, size=(3,)).astype(np.float32)
        eigvec = np.array(COCO_EIGEN_VALUES).astype(np.float32)
        eig_value = np.array(COCO_EIGEN_VECTORS).astype(np.float32)

        rgb_shift = eigvec @ (eig_value + alpha)
        img = image.astype(np.float32) + rgb_shift[None, None, :]
        return img

    return A.Compose(
        transforms=[
            A.RandomSizedCrop(
                min_max_height=(
                    int(train_patch_size[0] * 0.75),
                    int(train_patch_size[1] * 1.25),
                ),
                size=train_patch_size,
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.0,
                p=1.0,
            ),
            A.Lambda(image=lightning_),
            A.Normalize(mean=tuple(COCO_MEAN), std=tuple(COCO_STD)),
        ],
        bbox_params=A.BboxParams(
            format="coco",
            label_fields=["class_labels"],
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


class CocoTrainDataset(Dataset):
    def __init__(self, cfg: Config):

        self.data_dir = cfg.train_data_dir
        self.down_ratio = cfg.down_ratio
        self.patch_size = cfg.train_patch_size
        self.use_gaussian = cfg.use_gaussian
        self.gaussian_iou = cfg.gaussian_iou
        self.max_objs = cfg.max_objs
        self.n_classes = cfg.num_classes

        self._data_list = _get_data_list(str(self.data_dir), split="train")
        self._transform = _get_train_transform(self.patch_size)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, index):
        data = self._data_list[index]

        image_path = data["image"]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR_RGB)
        image = image.astype(np.uint8)
        bboxes = data["bboxes"]
        labels = data["labels"]

        _augmented = self._transform(image=image, bboxes=bboxes, class_labels=labels)
        image = _augmented["image"].transpose(2, 0, 1).astype(np.float32)
        bboxes = np.array(_augmented["bboxes"]).astype(np.float32)
        labels = np.array(_augmented["class_labels"]).astype(np.int32)

        bboxes[:, 2:] += bboxes[:, :2]  # xywh -> xyxy

        sorted_inds = np.argsort(labels, axis=0)
        bboxes = bboxes[sorted_inds]
        labels = labels[sorted_inds]

        feat_dict = self._create_training_feats(bboxes=bboxes, labels=labels)
        feat_dict["image"] = image
        return feat_dict

    def _create_training_feats(self, bboxes: np.ndarray, labels: np.ndarray):

        fmap_height, fmap_width = (
            (self.patch_size[0] + 1) // self.down_ratio,
            (self.patch_size[1] + 1) // self.down_ratio,
        )

        tl_hmap = np.zeros((self.n_classes, fmap_height, fmap_width), dtype=np.float32)
        br_hmap = np.zeros((self.n_classes, fmap_height, fmap_width), dtype=np.float32)
        ct_hmap = np.zeros((self.n_classes, fmap_height, fmap_width), dtype=np.float32)

        tl_regs = np.zeros((self.max_objs, 2), dtype=np.float32)
        br_regs = np.zeros((self.max_objs, 2), dtype=np.float32)
        ct_regs = np.zeros((self.max_objs, 2), dtype=np.float32)

        tl_inds = np.zeros((self.max_objs,), dtype=np.long)
        br_inds = np.zeros((self.max_objs,), dtype=np.long)
        ct_inds = np.zeros((self.max_objs,), dtype=np.long)

        # Marker mask to indicate valid objects out of max objs
        ind_masks = np.zeros((self.max_objs,), dtype=np.bool_)
        num_objs = min(bboxes.shape[0], self.max_objs)
        ind_masks[:num_objs] = 1

        for i in range(num_objs):
            bbox = bboxes[i]
            label = labels[i]

            tlx, tly, brx, bry = bbox / self.down_ratio
            ctx, cty = (tlx + brx) * 0.5, (tly + bry) * 0.5

            itlx, itly = int(tlx), int(tly)
            ibrx, ibry = int(brx), int(bry)
            ictx, icty = int(ctx), int(cty)

            tl_regs[i] = [tlx - itlx, tly - itly]
            br_regs[i] = [brx - ibrx, bry - ibry]
            ct_regs[i] = [ctx - ictx, cty - icty]

            if self.use_gaussian:
                box_width = math.ceil((bbox[2] - bbox[0]) / self.down_ratio)
                box_height = math.ceil((bbox[3] - bbox[1]) / self.down_ratio)

                radius = compute_gaussian_radius(box_height, box_width)

                tl_hmap[label] = draw_gaussian(tl_hmap[label], (itlx, itly), radius)
                br_hmap[label] = draw_gaussian(br_hmap[label], (ibrx, ibry), radius)
                ct_hmap[label] = draw_gaussian(ct_hmap[label], (ictx, icty), radius)

            else:
                tl_hmap[label, itly, itlx] = 1
                br_hmap[label, ibry, ibrx] = 1
                ct_hmap[label, ictx, icty] = 1

            tl_inds[i] = itly * fmap_width + itlx
            br_inds[i] = ibry * fmap_width + ibrx
            ct_inds[i] = icty * fmap_width + ictx

        return {
            "tl_hmap": tl_hmap,
            "br_hmap": br_hmap,
            "ct_hmap": ct_hmap,
            "tl_regs": tl_regs,
            "br_regs": br_regs,
            "ct_regs": ct_regs,
            "tl_inds": tl_inds,
            "br_inds": br_inds,
            "ct_inds": ct_inds,
            "ind_masks": ind_masks,
        }


class CocoValDataset(Dataset):
    def __init__(self, cfg: Config, device: torch.device):

        self.device = device
        self.data_dir = cfg.val_data_dir
        self.test_scales = cfg.test_scales
        self.down_ratio = cfg.down_ratio

        self._data_list = _get_data_list(str(self.data_dir), split="val")
        self._mean = torch.tensor(COCO_MEAN, device=device, dtype=torch.float)
        self._std = torch.tensor(COCO_STD, device=device, dtype=torch.float)
        
        self._initialize_cache(cache_rate=cfg.val_cache_rate)

    def _initialize_cache(self, cache_rate: float):
        """
        Initialize `_data_list` inplace by loading images. The number
        of items to load is determined by the given `cache_rate`.
        """
        n_cache = int(len(self._data_list) * cache_rate)
        paths = [d["image"] for d in self._data_list[:n_cache]]
        _load_func = lambda p: cv2.imread(p, cv2.IMREAD_COLOR_RGB)

        imgs = list(
            tqdm(
                Parallel(n_jobs=-1, return_as="generator")(
                    delayed(_load_func)(p) for p in paths
                ),
                total=len(paths),
                desc="Caching Validation images....",
            )
        )

        for i, img in enumerate(imgs):
            self._data_list[i]["image"] = img

    def __len__(self):
        return len(self._data_list)

    def _to_float_tensor(self, item):
        return torch.Tensor(item).to(device=self.device, dtype=torch.float)

    def __getitem__(self, index):
        data = self._data_list[index]
        image = data["image"]
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR_RGB)

        height, width = image.shape[:2]

        out_dict = {}

        for scale in self.test_scales:
            scaled_height = int(height * scale)
            scaled_width = int(width * scale)

            resized_img = cv2.resize(image, (scaled_width, scaled_height))
            resized_img = torch.from_numpy(resized_img).permute(2, 0, 1)
            resized_img = resized_img.to(
                dtype=torch.float, device=self.device, non_blocking=True
            )

            test_height = scaled_height | 127
            test_width = scaled_width | 127

            fmap_height, fmap_width = (
                (test_height + 1) // self.down_ratio,
                (test_width + 1) // self.down_ratio,
            )
            ratio_height, ratio_width = (
                fmap_height / test_height,
                fmap_width / test_width,
            )
            test_img, border, _ = center_crop_gpu(
                resized_img, new_size=(test_height, test_width)
            )

            test_img /= 255.0
            test_img -= self._mean[:, None, None]
            test_img /= self._std[:, None, None]

            out_dict[scale] = {
                "image": test_img,
                "borders": border,
                "size": self._to_float_tensor((scaled_height, scaled_width)),
                "fmap_size": self._to_float_tensor((fmap_height, fmap_width)),
                "ratios": self._to_float_tensor((ratio_height, ratio_width)),
            }
            
        return data["image_id"], out_dict
