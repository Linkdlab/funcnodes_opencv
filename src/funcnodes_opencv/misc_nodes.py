import numpy as np
import cv2
import funcnodes as fn
from typing import Literal
from .imageformat import OpenCVImageFormat, ImageFormat
from .utils import assert_opencvdata


@fn.NodeDecorator(
    node_id="cv2.replace_channel",
    name="Replace Channel",
    description="Replace a channel in the image with another images channel.",
    default_render_options={"data": {"src": "out"}},
)
def replace_channel(
    trg_img: ImageFormat,
    src_img: ImageFormat,
    channel: Literal[1, 2, 3] = 1,
) -> OpenCVImageFormat:
    """
    Replace a channel in the image with another image.
    :param trg_img: The target image.
    :param src_img: The source image.
    :param channel: The channel to replace.
    :return: The modified image.
    """
    channel = int(channel) - 1
    trg_data = assert_opencvdata(trg_img, channel=3)
    src_data = assert_opencvdata(src_img)
    trg_data[:, :, channel] = (
        src_data[:, :, channel] if src_data.shape[2] == 3 else src_data[:, :, 0]
    )

    return OpenCVImageFormat(trg_data)


@fn.NodeDecorator(
    node_id="cv2.minmax_lcn",
    name="MinMax LCN",
    description="Performs Local Contrast Normalization using Min-Max method.",
    default_render_options={"data": {"src": "out"}},
)
def minmax_lcn(
    img: ImageFormat, ksize: int = 8, mincontrast: int = 0, clip: bool = True
) -> OpenCVImageFormat:
    img = assert_opencvdata(img, channel=3)
    kernel = np.ones((ksize, ksize), np.uint8)
    upp = cv2.dilate(img, kernel)
    low = cv2.erode(img, kernel)
    upp = cv2.blur(upp, ksize=(ksize, ksize))  # faster
    low = cv2.blur(low, ksize=(ksize, ksize))
    contrast = np.maximum(upp - low, mincontrast)
    res = (img - low) / contrast
    if clip:
        res = np.clip(res, 0, 1)
    return OpenCVImageFormat(res)


NODE_SHELF = fn.Shelf(
    name="Misc Nodes",
    nodes=[
        replace_channel,
        minmax_lcn,
    ],
    description="Miscellaneous nodes for image processing",
    subshelves=[],
)
