import math

import numpy as np
import torch


def compute_gaussian_radius(box_height, box_width, min_overlap=0.7) -> int:
    # solve 3 quadratic equations for radius
    a1 = 1
    b1 = box_height + box_width
    c1 = box_width * box_height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = math.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 - sq1) / (2 * a1)

    a2 = 4
    b2 = 2 * (box_height + box_width)
    c2 = (1 - min_overlap) * box_width * box_height
    sq2 = math.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 - sq2) / (2 * a2)

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (box_height + box_width)
    c3 = (min_overlap - 1) * box_width * box_height
    sq3 = math.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / (2 * a3)
    return max(0, int(min(r1, r2, r3)))


def gaussian_2d(shape: tuple[int, int], sigma: float = 1.0):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(hmap: np.ndarray, center: tuple[int, int], radius: int):
    x, y = center
    height, width = hmap.shape

    diameter = 2 * radius + 1
    kernel = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x_slice = slice(max(0, x - radius), min(width, x + radius + 1))
    y_slice = slice(max(0, y - radius), min(height, y + radius + 1))

    ky_min = max(0, radius - y)
    kx_min = max(0, radius - x)
    ky_max = ky_min + (y_slice.stop - y_slice.start)
    kx_max = kx_min + (x_slice.stop - x_slice.start)
    ky_slice = slice(ky_min, ky_max)
    kx_slice = slice(kx_min, kx_max)

    hmap[y_slice, x_slice] = np.maximum(
        hmap[y_slice, x_slice], kernel[ky_slice, kx_slice]
    )

    return hmap


def center_crop_gpu(image: torch.Tensor, new_size: tuple[int, int]):

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
        dtype=torch.float,
    )

    # offset of the top-left valid region without clipping
    offset = image.new_tensor([im_cty - new_height // 2, im_ctx - new_width // 2])

    return new_image, border, offset
