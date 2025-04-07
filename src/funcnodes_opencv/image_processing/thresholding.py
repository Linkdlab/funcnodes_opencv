from typing import Tuple
import cv2
import numpy as np
import funcnodes as fn
from ..imageformat import OpenCVImageFormat, ImageFormat
from ..utils import assert_opencvdata


class ThresholdTypes(fn.DataEnum):
    """
    Threshold types.

    Attributes:
        BINARY: cv2.THRESH_BINARY: 0 or maxval (if x > thresh)
        BINARY_INV: cv2.THRESH_BINARY_INV: maxval or 0 (if x > thresh)
        TRUNC: cv2.THRESH_TRUNC: thresh or x (if x > thresh)
        TOZERO: cv2.THRESH_TOZERO: x or 0 (if x > thresh)
        TOZERO_INV: cv2.THRESH_TOZERO_INV 0 or x (if x > thresh)
    """

    BINARY = cv2.THRESH_BINARY
    BINARY_INV = cv2.THRESH_BINARY_INV
    TRUNC = cv2.THRESH_TRUNC
    TOZERO = cv2.THRESH_TOZERO
    TOZERO_INV = cv2.THRESH_TOZERO_INV


@fn.NodeDecorator(
    node_id="cv2.threshold",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
    ],
    default_io_options={
        "maxval": {"value_options": {"min": 0.0, "max": 1.0}},
        "thresh": {"value_options": {"min": 0.0, "max": 1.0}},
    },
    default_render_options={"data": {"src": "out"}},
    description="Apply a fixed-level threshold to an image.",
)
def threshold(
    img: ImageFormat,
    thresh: float = 0,
    maxval: float = 1.0,
    type: ThresholdTypes = ThresholdTypes.BINARY,
) -> OpenCVImageFormat:
    return OpenCVImageFormat(
        cv2.threshold(assert_opencvdata(img), thresh, maxval, ThresholdTypes.v(type))[1]
    )


class AutoThresholdTypes(fn.DataEnum):
    """
    OTSU = cv2.THRESH_OTSU : Otsu's thresholding
    TRIANGLE = cv2.THRESH_TRIANGLE: Triangle thresholding
    """

    OTSU = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    TRIANGLE = cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE


@fn.NodeDecorator(
    node_id="cv2.auto_threshold",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
        {"name": "thresh", "type": int},
    ],
    default_io_options={"maxval": {"value_options": {"min": 0.0, "max": 1.0}}},
    default_render_options={"data": {"src": "out"}},
    description="Apply an automatic threshold to an image.",
)
def auto_threshold(
    img: ImageFormat,
    maxval: float = 1,
    type: AutoThresholdTypes = AutoThresholdTypes.OTSU,
) -> Tuple[OpenCVImageFormat, int]:
    type = AutoThresholdTypes.v(type)

    img = assert_opencvdata(img, 1)

    if type == AutoThresholdTypes.OTSU.value:
        img = (img * 65535).astype(np.uint16)
        maxval = int(maxval * 65535)
    elif type == AutoThresholdTypes.TRIANGLE.value:
        img = (img * 255).astype(np.uint8)
        maxval = int(maxval * 255)

    thresh, img = cv2.threshold(img, 0, maxval, type)
    return OpenCVImageFormat(img), thresh


class AdaptiveThresholdMethods(fn.DataEnum):
    """
    Adaptive threshold methods.

    Attributes:
        MEAN_C: cv2.ADAPTIVE_THRESH_MEAN_C: threshold value is the mean of the neighbourhood area
        GAUSSIAN_C: cv2.ADAPTIVE_THRESH_GAUSSIAN_C: threshold value is the weighted sum of the
            neighbourhood values where weights are a gaussian window
    """

    MEAN_C = cv2.ADAPTIVE_THRESH_MEAN_C
    GAUSSIAN_C = cv2.ADAPTIVE_THRESH_GAUSSIAN_C


@fn.NodeDecorator(
    node_id="cv2.adaptive_threshold",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
    ],
    default_io_options={
        "maxval": {"value_options": {"min": 0.0, "max": 1.0}},
        "c": {"value_options": {"min": -1, "max": 1}},
        "block_size": {"value_options": {"min": 1}},
    },
    default_render_options={"data": {"src": "out"}},
    description="Apply an adaptive threshold to an image.",
)
def adaptive_threshold(
    img: ImageFormat,
    maxval: float = 1,
    method: AdaptiveThresholdMethods = AdaptiveThresholdMethods.MEAN_C,
    block_size: int = 1,
    c: float = 0,
) -> OpenCVImageFormat:
    # threshold_type = ThresholdTypes.v(threshold_type)
    block_size = 2 * int(block_size) + 1
    img = assert_opencvdata(img, 1)
    img = (img * 255).astype(np.uint8)
    maxval = int(maxval * 255)

    return OpenCVImageFormat(
        cv2.adaptiveThreshold(
            img,
            maxval,
            AdaptiveThresholdMethods.v(method),
            cv2.THRESH_BINARY,  # fixed because not all types are allowed
            block_size,
            c * 255,
        )
    )


@fn.NodeDecorator(
    node_id="cv2.in_range_sc",
    node_name=" In Range Single Channel",
    default_io_options={
        "lower": {"value_options": {"min": 0.0, "max": 1.0}},
        "upper": {"value_options": {"min": 0.0, "max": 1.0}},
    },
    default_render_options={"data": {"src": "out"}},
)
def in_range_singel_channel(
    img: ImageFormat, lower: float = 0, upper: float = 1
) -> OpenCVImageFormat:
    return OpenCVImageFormat(
        cv2.inRange(
            assert_opencvdata(img, channel=1),
            np.array([lower], dtype=np.float32),
            np.array([upper], dtype=np.float32),
        )
    )


@fn.NodeDecorator(
    node_id="cv2.in_range",
    node_name="In Range",
    default_io_options={
        "lower_c1": {"value_options": {"min": 0.0, "max": 1.0}},
        "upper_c1": {"value_options": {"min": 0.0, "max": 1.0}},
        "lower_c2": {"value_options": {"min": 0.0, "max": 1.0}},
        "upper_c2": {"value_options": {"min": 0.0, "max": 1.0}},
        "lower_c3": {"value_options": {"min": 0.0, "max": 1.0}},
        "upper_c3": {"value_options": {"min": 0.0, "max": 1.0}},
    },
    default_render_options={"data": {"src": "out"}},
)
def in_range(
    img: ImageFormat,
    lower_c1: float = 0,
    upper_c1: float = 1,
    lower_c2: float = 0,
    upper_c2: float = 1,
    lower_c3: float = 0,
    upper_c3: float = 1,
) -> OpenCVImageFormat:
    data = assert_opencvdata(img)
    if data.shape[2] == 1:
        arr = cv2.inRange(
            data,
            np.array([lower_c1], dtype=np.float32),
            np.array([upper_c1], dtype=np.float32),
        )
    else:
        arr = cv2.inRange(
            data,
            np.array([lower_c1, lower_c2, lower_c3], dtype=np.float32),
            np.array([upper_c1, upper_c2, upper_c3], dtype=np.float32),
        )

    return OpenCVImageFormat(arr)


NODE_SHELF = fn.Shelf(
    name="Masking and Thresholding",
    description="OpenCV image masking and thresholding nodes.",
    subshelves=[],
    nodes=[
        threshold,
        auto_threshold,
        adaptive_threshold,
        in_range_singel_channel,
        in_range,
    ],
)
