import cv2
import funcnodes as fn

import numpy as np
from ..imageformat import OpenCVImageFormat, ImageFormat
from ..utils import assert_opencvdata, assert_similar_opencvdata


@fn.NodeDecorator(
    node_id="cv2.bitwise_and",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Bitwise AND operation on two images.",
)
def bitwise_and(
    img1: ImageFormat,
    img2: ImageFormat,
    mask: ImageFormat = None,
) -> OpenCVImageFormat:
    data1, data2 = assert_similar_opencvdata(img1, img2)
    data1 = (data1 * 65535).astype(np.uint16)
    data2 = (data2 * 65535).astype(np.uint16)
    mask = assert_opencvdata(mask, channel=1) if mask is not None else None
    mask = (mask * 255).astype(np.uint8) if mask is not None else None
    result = cv2.bitwise_and(data1, data2, mask=mask)
    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.bitwise_or",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
)
def bitwise_or(
    img1: ImageFormat,
    img2: ImageFormat,
    mask: ImageFormat = None,
) -> OpenCVImageFormat:
    data1, data2 = assert_similar_opencvdata(img1, img2)
    data1 = (data1 * 65535).astype(np.uint16)
    data2 = (data2 * 65535).astype(np.uint16)
    mask = assert_opencvdata(mask, channel=1) if mask is not None else None
    mask = (mask * 255).astype(np.uint8) if mask is not None else None
    result = cv2.bitwise_or(data1, data2, mask=mask)
    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.bitwise_xor",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Bitwise XOR operation on two images.",
)
def bitwise_xor(
    img1: ImageFormat,
    img2: ImageFormat,
    mask: ImageFormat = None,
) -> OpenCVImageFormat:
    data1, data2 = assert_similar_opencvdata(img1, img2)
    data1 = (data1 * 65535).astype(np.uint16)
    data2 = (data2 * 65535).astype(np.uint16)
    mask = assert_opencvdata(mask, channel=1) if mask is not None else None
    mask = (mask * 255).astype(np.uint8) if mask is not None else None
    result = cv2.bitwise_xor(data1, data2, mask=mask)
    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.bitwise_not",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Bitwise NOT operation on an image.",
)
def bitwise_not(
    img: ImageFormat,
    mask: ImageFormat = None,
) -> OpenCVImageFormat:
    data = assert_opencvdata(img)
    data = (data * 65535).astype(np.uint16)
    mask = assert_opencvdata(mask, channel=1) if mask is not None else None
    mask = (mask * 255).astype(np.uint8) if mask is not None else None
    result = cv2.bitwise_not(data, mask=mask)
    return OpenCVImageFormat(result)


NODE_SHELF = fn.Shelf(
    name="Bitwise Operations",
    description="OpenCV bitwise operations on images.",
    subshelves=[],
    nodes=[bitwise_and, bitwise_or, bitwise_xor, bitwise_not],
)
