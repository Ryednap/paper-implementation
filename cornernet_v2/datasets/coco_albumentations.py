import math
import cv2
import numpy as np
import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Tuple, cast
import albumentations as A

import monai.data as md
import monai.transforms as mt
import monai.apps.detection.transforms.dictionary as mt_det

import torch
from torch.utils.data import Dataset

from configs.base import Config
from ._coco_constants import (
    COCO_NAMES,
    COCO_IDS,
    COCO_MEAN,
    COCO_STD,
    COCO_EIGEN_VALUES,
    COCO_EIGEN_VECTORS,
)
from .serialize import TorchSerializedList

cv2.setNumThreads(0)


def _get_data_list(data_dir: Path, split: Literal["train", "val", "test"]):
    image_dir = data_dir / "images" / f"{split}2014"
    annot_path = data_dir / "annotations" / f"instances_{split}2014.json"

    coco_gt = coco.COCO(annot_path.as_posix())
    image_ids = coco_gt.getImgIds()

    data_list = []
    for img_id in image_ids:
        image_path = image_dir / coco_gt.loadImgs(img_id)[0]["file_name"]
        annotations = coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=img_id))

        labels = [anno["category_id"] for anno in annotations]
        labels = np.array([COCO_IDS.index(label) for label in labels])
        bboxes = np.array([anno["bbox"] for anno in annotations])

        data_list.append(
            {
                "image_id": img_id,
                "image": image_path.as_posix(),
                "bboxes": bboxes,
                "labels": labels,
            }
        )

    return TorchSerializedList(data_list)


def _get_train_transform(cfg: Config):

    def lightning_(image, alpha_std: float, rng=np.random):
        alpha = rng.normal(scale=alpha_std, size=(3,)).astype(np.float32)
        eigvec = np.array(COCO_EIGEN_VALUES).astype(np.float32)
        eig_value = np.array(COCO_EIGEN_VECTORS).astype(np.float32)

        rgb_shift = (eigvec @ (eig_value + alpha)).astype(np.float32)
        img = image.astype(np.float32) + rgb_shift[None, None, :]
        return img

    return A.Compose(
        transforms=[
            A.RandomResizedCrop(
                size=cfg.train_patch_size,
                scale=(0.75, 1.25),
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
            min_visibility=0.1,
            clip=True,
            filter_invalid_bboxes=True,
        ),
    )


class CocoTrainDataset(Dataset):
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg

        self.data_list = _get_data_list(cfg.data_dir, "train")
        self.transform = _get_train_transform(cfg=cfg)

        self._num_classes = len(COCO_NAMES) - 1
        self._max_objs = 128
        self._fmap_size = {
            "h": (cfg.train_patch_size[0] + 1) // cfg.down_ratio,
            "w": (cfg.train_patch_size[1] + 1) // cfg.down_ratio,
        }

    def __len__(self):
        len(self.data_list)

    @staticmethod
    def compute_gaussian_radius(box_height, box_width, min_overlap=0.7) -> int:
        # solve 3 quadratic equations for radius
        a1 = 1
        b1 = box_height + box_width
        c1 = box_width * box_height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = math.sqrt(b1**2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / 2

        a2 = 4
        b2 = 2 * (box_height + box_width)
        c2 = (1 - min_overlap) * box_width * box_height
        sq2 = math.sqrt(b2**2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / 2

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (box_height + box_width)
        c3 = (min_overlap - 1) * box_width * box_height
        sq3 = math.sqrt(b3**2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2

        return int(min(r1, r2, r3))

    @staticmethod
    def _gaussian_2d(shape: Tuple[int, int], sigma: float = 1.0):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    @staticmethod
    def draw_gaussian(
        hmap: torch.Tensor, center: Tuple[int, int], radius: int
    ) -> torch.Tensor:
        diameter = 2 * radius + 1
        kernel = torch.from_numpy(
            CocoTrainDataset._gaussian_2d((diameter, diameter), sigma=diameter / 6)
        ).to(hmap.device)

        x, y = center

        height, width = hmap.shape[0:2]

        y_slice = slice(max(0, y - radius), min(height, y + radius + 1))
        x_slice = slice(max(0, x - radius), min(width, x + radius + 1))

        ky_min = max(0, radius - y)
        ky_max = ky_min + (y_slice.stop - y_slice.start)

        kx_min = max(0, radius - x)
        kx_max = kx_min + (x_slice.stop - x_slice.start)

        hmap[y_slice, x_slice] = torch.maximum(
            hmap[y_slice, x_slice], kernel[ky_min:ky_max, kx_min:kx_max]
        )
        return hmap

    def __getitem__(self, index):
        data = self.data_list[index]

        image = cv2.imread(data["image"], cv2.IMREAD_COLOR_RGB).astype(np.uint8)
        bboxes = data["bboxes"]
        labels = data["labels"]

        augmented = self.transform(image=image, bboxes=bboxes, class_labels=labels)
        image = torch.from_numpy(augmented["image"]).to(dtype=torch.float32)
        bboxes = torch.from_numpy(augmented["bboxes"]).to(dtype=torch.float32)
        labels = torch.from_numpy(augmented["class_labels"]).to(dtype=torch.long)

        bboxes[:, 2:] += bboxes[:, :2]  # xywh -> xyxy

        sorted_inds = torch.argsort(labels, dim=0)
        bboxes = bboxes[sorted_inds]
        labels = labels[sorted_inds]

        # top left features
        tl_hmap = torch.zeros(
            (self._num_classes, self._fmap_size["h"], self._fmap_size["w"]),
            dtype=torch.float32,
        )
        tl_regs = torch.zeros((self._max_objs, 2), dtype=torch.float32)
        tl_indices = torch.zeros((self._max_objs,), dtype=torch.int64)

        # bottom right features
        br_hmap = torch.zeros(
            (self._num_classes, self._fmap_size["h"], self._fmap_size["w"]),
            dtype=torch.float32,
        )
        br_regs = torch.zeros((self._max_objs, 2), dtype=torch.float32)
        br_indices = torch.zeros((self._max_objs,), dtype=torch.int64)

        # Marker masks to indicate valid objects out of max_objs
        ind_masks = torch.zeros((self._max_objs,), dtype=torch.uint8)
        num_objs = min(bboxes.shape[0], self._max_objs)
        ind_masks[:num_objs] = 1

        for i in range(num_objs):
            bbox = bboxes[i]
            cls_id = labels[i]

            tl_x, tl_y, br_x, br_y = bbox / self.cfg.down_ratio

            tl_x_int = int(tl_x)
            tl_y_int = int(tl_y)
            br_x_int = int(br_x)
            br_y_int = int(br_y)

            if self.cfg.use_gaussian:
                box_width = math.ceil((bbox[2] - bbox[0]) / self.cfg.down_ratio)
                box_height = math.ceil((bbox[3] - bbox[1]) / self.cfg.down_ratio)
                radius = self.compute_gaussian_radius(
                    box_height, box_width, min_overlap=self.cfg.gaussian_iou
                )

                tl_hmap[cls_id] = self.draw_gaussian(
                    tl_hmap[cls_id], (tl_x_int, tl_y_int), radius
                )
                br_hmap[cls_id] = self.draw_gaussian(
                    br_hmap[cls_id], (br_x_int, br_y_int), radius
                )
            else:
                # hard label assignment
                tl_hmap[cls_id, tl_y_int, tl_x_int] = 1
                br_hmap[cls_id, br_y_int, br_x_int] = 1

            tl_regs[i] = torch.tensor(
                [tl_x - tl_x_int, tl_y - tl_y_int],
                dtype=torch.float32,
            )
            br_regs[i] = torch.tensor(
                [br_x - br_x_int, br_y - br_y_int],
                dtype=torch.float32,
            )

            # flattened indices which indicates the box position in feature map
            # example if we have 2D array of size (h, w)
            # the flattened index is y * w + x
            tl_indices[i] = tl_y_int * self._fmap_size["w"] + tl_x_int
            br_indices[i] = br_y_int * self._fmap_size["w"] + br_x_int

        return {
            "image": image,
            "tl_hmap": tl_hmap,
            "tl_regs": tl_regs,
            "tl_indices": tl_indices,
            "br_hmap": br_hmap,
            "br_regs": br_regs,
            "br_indices": br_indices,
            "ind_masks": ind_masks,
        }


def _center_crop(image: torch.Tensor, new_size: Tuple[int, int]):

    im_height, im_width = image.shape[1:]
    im_ctx, im_cty = im_width // 2, im_height // 2

    new_height, new_width = new_size
    new_ctx, new_cty = new_width // 2, new_height // 2

    new_image = image.new_zeros((3, new_height, new_width))

    # compute valid region of the original region which needs to be present
    # in the new cropped image. If new size is large then entire image
    # else we need to compute how much region from image center to take
    valid_x0, valid_y0 = (
        max(0, im_ctx - new_width // 2),
        max(0, im_cty - new_height // 2),
    )
    valid_x1, valid_y1 = (
        min(im_width, im_ctx + new_width // 2),
        min(im_height, im_cty + new_height // 2),
    )

    # Now assume the new image as the canvas we need to place the valid region
    # of the original image to the convas such that we maintain the center.
    # Hence, we need to compute how much we need to move dx, dy from canvas
    # center in left, right, bottom and top directions
    left_dx, right_dx = im_ctx - valid_x0, valid_x1 - im_ctx
    top_dy, bottom_dy = im_cty - valid_y0, valid_y1 - im_cty

    # slice region in the new canvas to paint
    x_slice = slice(new_ctx - left_dx, new_ctx + right_dx)
    y_slice = slice(new_cty - top_dy, new_cty + bottom_dy)

    # paint the canvas
    new_image[:, y_slice, x_slice] = image[:, valid_y0:valid_y1, valid_x0:valid_x1]

    # top-left and bottom right coordinates in new image
    # which contains our original image
    border = image.new_tensor(
        [new_cty - top_dy, new_cty + bottom_dy, new_ctx - left_dx, new_ctx + right_dx],
        dtype=torch.float32,
    )

    # offset of the top-left valid region without clipping
    offset = image.new_tensor([im_cty - new_height // 2, im_ctx - new_width // 2])

    return new_image, border, offset


class CocoValDataset(Dataset):
    def __init__(self, cfg: Config, device: torch.device):
        self.cfg = cfg
        self.device = "cpu"
        self.test_scales = cfg.test_scales

        self.data_list = _get_data_list(cfg.val_data_dir, "val")

        self._mean = torch.tensor(COCO_MEAN, dtype=torch.float32, device=self.device)
        self._std = torch.tensor(COCO_STD, dtype=torch.float32, device=self.device)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]

        image = cast(np.ndarray, cv2.imread(data["image"], cv2.IMREAD_COLOR))
        try:
            height, width = image.shape[0:2]
        except:
            raise

        out_dict = {}

        for scale in self.test_scales:
            scaled_height = int(height * scale)
            scaled_width = int(width * scale)

            resized_image = cv2.resize(image, (scaled_width, scaled_height))
            resized_image = torch.from_numpy(resized_image).permute(2, 0, 1)
            resized_image = resized_image.to(dtype=torch.float32, device=self.device)

            target_height = scaled_height | 127
            target_width = scaled_width | 127

            fmap_height, fmap_width = (
                (target_height + 1) // self.cfg.down_ratio,
                (target_width + 1) // self.cfg.down_ratio,
            )
            target_image, border, _ = _center_crop(
                image=resized_image, new_size=(target_height, target_width)
            )

            target_image /= 255.0
            target_image -= self._mean[:, None, None]
            target_image /= self._std[:, None, None]

            out_dict[scale] = {
                "image": target_image,
                "border": border,
                "original_size": torch.Tensor((scaled_height, scaled_width)).to(
                    device=self.device, dtype=torch.float32
                ),
                "fmap_size": torch.Tensor((fmap_height, fmap_width)).to(
                    device=self.device, dtype=torch.float32
                ),
                "ratio": torch.Tensor(
                    [fmap_height / target_height, fmap_width / target_width]
                ).to(device=self.device, dtype=torch.float32),
            }

        return data["image_id"], out_dict
