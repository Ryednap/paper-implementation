from typing import Tuple, cast
from monai.apps.detection.transforms.array import (
    clip_boxes_to_image,
    convert_box_to_mask,
)
from monai.apps.detection.transforms.dictionary import KeysCollection, convert_data_type
from monai.transforms import MapTransform
import torch


class GenerateExtendedBoxMask(MapTransform):
    """
    Generate Box mask on the input box
    """

    def __init__(
        self,
        keys: KeysCollection,
        image_key: str,
        spatial_size: Tuple[int, int],
        whole_box: bool,
        mask_image_key: str = "mask_image",
    ):
        super().__init__(keys)
        self.image_key = image_key
        self.spatial_size = spatial_size
        self.whole_box = whole_box
        self.mask_image_key = mask_image_key

    def generate_fg_center_boxes(self, boxes, image_size, whole_box=True):
        # We don't require crop center to be within the boxes.
        # As along as the cropped patch contains a box, it is considered as a foreground patch.
        # Positions within extended_boxes are crop centers for foreground patches
        spatial_dims = len(image_size)
        boxes_torch, *_ = convert_data_type(boxes, torch.Tensor)

        extended_boxes = torch.zeros_like(boxes_torch, dtype=torch.int)
        boxes_start = torch.ceil(
            boxes_torch[:, :spatial_dims]
        )  # (x1, y1) / (x1, y1, z1)
        boxes_end = torch.floor(
            boxes_torch[:, spatial_dims:]
        )  # (x2, y2) / (x2, y2, z2)

        for axis in range(spatial_dims):
            if not whole_box:
                extended_boxes[:, axis] = (
                    boxes_start[:, axis] - self.spatial_size[axis] // 2 + 1
                )
                extended_boxes[:, axis + spatial_dims] = (
                    boxes_end[:, axis] + self.spatial_size[axis] // 2 - 1
                )
            else:
                # extended box start
                extended_boxes[:, axis] = (
                    boxes_end[:, axis] - self.spatial_size[axis] // 2 - 1
                )
                extended_boxes[:, axis] = torch.minimum(
                    extended_boxes[:, axis], boxes_start[:, axis]
                )
                # extended box end
                extended_boxes[:, axis + spatial_dims] = (
                    boxes_start[:, axis] + self.spatial_size[axis] // 2 + 1
                )
                extended_boxes[:, axis + spatial_dims] = torch.maximum(
                    extended_boxes[:, axis + spatial_dims], boxes_end[:, axis]
                )

        extended_boxes, _ = clip_boxes_to_image(
            extended_boxes, image_size, remove_empty=True
        )
        return extended_boxes

    def generate_mask_img(self, boxes, image_size, whole_box=True):
        extended_boxes = self.generate_fg_center_boxes(boxes, image_size, whole_box)
        mask_img = cast(
            torch.Tensor,
            convert_box_to_mask(
                extended_boxes,
                labels=torch.ones(extended_boxes.shape[0]),
                spatial_size=image_size,
                bg_label=0,
                ellipse_mask=True,
            ),
        )
        mask_img = torch.amax(mask_img, dim=0, keepdim=True)[0:1, ...]
        return mask_img

    def __call__(self, data):
        d = dict(data)
        for key in self.key_iterator(d):
            image = d[self.image_key]
            boxes = d[key]
            data[self.mask_image_key] = self.generate_mask_img(boxes, image.shape[1:])
        return data