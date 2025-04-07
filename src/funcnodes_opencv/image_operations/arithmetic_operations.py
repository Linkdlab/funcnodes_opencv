from typing import Union
import cv2
import numpy as np
import funcnodes as fn
from ..imageformat import OpenCVImageFormat, ImageFormat
from ..utils import assert_opencvdata, assert_similar_opencvdata


@fn.NodeDecorator(
    node_id="cv2.add",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Add two images by adding their pixel values.",
)
def add(
    img1: ImageFormat,
    img2: ImageFormat,
    mask: ImageFormat = None,
) -> OpenCVImageFormat:
    data1, data2 = assert_similar_opencvdata(img1, img2)
    mask = assert_opencvdata(mask, channel=1) if mask is not None else None
    result = cv2.add(data1, data2, mask=mask)
    result = np.clip(result, a_min=0.0, a_max=1.0)

    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.subtract",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Subtract two images by subtracting their pixel values.",
)
def subtract(
    img1: ImageFormat,
    img2: ImageFormat,
    mask: ImageFormat = None,
) -> OpenCVImageFormat:
    data1, data2 = assert_similar_opencvdata(img1, img2)
    mask = assert_opencvdata(mask, channel=1) if mask is not None else None
    result = cv2.subtract(data1, data2, mask=mask)
    result = np.clip(result, a_min=0.0, a_max=1.0)
    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.multiply",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Multiply two images by multiplying their pixel values.",
)
def multiply(
    img1: ImageFormat,
    img2: ImageFormat,
) -> OpenCVImageFormat:
    data1, data2 = assert_similar_opencvdata(img1, img2)
    result = cv2.multiply(
        data1,
        data2,
    )
    result = np.clip(result, a_min=0.0, a_max=1.0)
    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.divide",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Divide two images by dividing their pixel values.",
)
def divide(
    img1: ImageFormat,
    img2: ImageFormat,
) -> OpenCVImageFormat:
    data1, data2 = assert_similar_opencvdata(img1, img2)
    result = cv2.divide(data1, data2 + 1e-16)  # Avoid division by zero
    result = np.clip(result, a_min=0.0, a_max=1.0)

    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.addWeighted",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Add two images by adding their pixel values with a specified weight.",
    default_io_options={
        "ratio": {"value_options": {"min": 0.0, "max": 1.0}},
    },
)
def addWeighted(
    img1: ImageFormat,
    img2: ImageFormat,
    ratio: float = 0.5,
) -> OpenCVImageFormat:
    data1, data2 = assert_similar_opencvdata(img1, img2)
    alpha = max(0, min(float(ratio), 1))
    beta = 1.0 - alpha
    result = cv2.addWeighted(data1, alpha, data2, beta, 0)
    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.where",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    default_io_options={
        "value": {"value_options": {"min": 0.0, "max": 1.0}},
    },
    description="Based on a mask sets the pixel values of an image to a specified value or another image.",
)
def where(
    img: ImageFormat,
    mask: ImageFormat,
    value: Union[float, ImageFormat],
) -> OpenCVImageFormat:
    if isinstance(value, (int, float)):
        data, mask = assert_similar_opencvdata(img, mask)
    else:
        data, mask, value = assert_similar_opencvdata(img, mask, value)
    mask = mask.astype(bool)

    result = data.copy()
    result[mask] = value if isinstance(value, (int, float)) else value[mask]
    return OpenCVImageFormat(result)


@fn.NodeDecorator(
    node_id="cv2.brighten",
    outputs=[{"name": "out", "type": OpenCVImageFormat}],
    default_render_options={"data": {"src": "out"}},
    description="Brighten an image by adding a specified value to all pixel values.",
    default_io_options={"value": {"value_options": {"min": -1, "max": 1}}},
)
def brighten(
    img: ImageFormat,
    value: float = 0,
) -> OpenCVImageFormat:
    data = assert_opencvdata(img)
    result = np.clip(data + value, a_min=0.0, a_max=1.0)
    return OpenCVImageFormat(result)


NODE_SHELF = fn.Shelf(
    name="Arithmetic Operations",
    nodes=[add, subtract, multiply, divide, addWeighted, where],
    description="Nodes for performing arithmetic operations on images.",
    subshelves=[],
)
