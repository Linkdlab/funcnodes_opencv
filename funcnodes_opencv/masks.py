from typing import Tuple, Literal, Union
import cv2
import numpy as np
from .imageformat import OpenCVImageFormat, ImageFormat, NumpyImageFormat
import funcnodes as fn
from .utils import assert_opencvdata


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


class AutoThresholdTypes(fn.DataEnum):
    """
    OTSU = cv2.THRESH_OTSU : Otsu's thresholding
    TRIANGLE = cv2.THRESH_TRIANGLE: Triangle thresholding
    """

    OTSU = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    TRIANGLE = cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE


@fn.NodeDecorator(
    node_id="cv2.threshold",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
    ],
    default_io_options={
        "maxval": {"value_options": {"min": 0, "max": 255}},
        "thresh": {"value_options": {"min": 0, "max": 255}},
    },
    default_render_options={"data": {"src": "out"}},
)
def threshold(
    img: ImageFormat,
    thresh: int = 0,
    maxval: int = 255,
    type: ThresholdTypes = ThresholdTypes.BINARY,
) -> OpenCVImageFormat:
    type = ThresholdTypes.v(type)
    return OpenCVImageFormat(
        cv2.threshold(assert_opencvdata(img), thresh, maxval, type)[1]
    )


@fn.NodeDecorator(
    node_id="cv2.auto_threshold",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
        {"name": "thresh", "type": int},
    ],
    default_io_options={"maxval": {"value_options": {"min": 0, "max": 255}}},
    default_render_options={"data": {"src": "out"}},
)
def auto_threshold(
    img: ImageFormat,
    maxval: int = 255,
    type: AutoThresholdTypes = AutoThresholdTypes.OTSU,
) -> Tuple[OpenCVImageFormat, int]:
    type = AutoThresholdTypes.v(type)
    thresh, img = cv2.threshold(assert_opencvdata(img, 1), 0, maxval, type)
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
        "maxval": {"value_options": {"min": 0, "max": 255}},
        "c": {"value_options": {"min": -255, "max": 255}},
        "block_size": {"value_options": {"min": 1}},
    },
    default_render_options={"data": {"src": "out"}},
)
def adaptive_threshold(
    img: ImageFormat,
    maxval: int = 255,
    adaptive_method: AdaptiveThresholdMethods = AdaptiveThresholdMethods.MEAN_C,
    block_size: int = 1,
    c: int = 0,
) -> OpenCVImageFormat:
    adaptive_method = AdaptiveThresholdMethods.v(adaptive_method)
    # threshold_type = ThresholdTypes.v(threshold_type)
    block_size = 2 * int(block_size) + 1
    return OpenCVImageFormat(
        cv2.adaptiveThreshold(
            assert_opencvdata(img, 1),
            maxval,
            adaptive_method,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )
    )


class DistanceTypes(fn.DataEnum):
    """
    Distance transform types.

    Attributes:
        L1: cv2.DIST_L1: distance = |x1-x2| + |y1-y2|
        L2: cv2.DIST_L2: the Euclidean distance
        C: cv2.DIST_C: distance = max(|x1-x2|, |y1-y2|)
        L12: cv2.DIST_L12: L1-L2 metric: distance = 2 * (sqrt(1 + |x1-x2|^2/2) - 1)
        FAIR: cv2.DIST_FAIR: distance = c^2 * (sqrt(1 + |x1-x2|^2/c^2) - 1)
        WELSCH: cv2.DIST_WELSCH: distance = c^2/2 * (1 - exp(-|x1-x2|^2/c^2))
        HUBER: cv2.DIST_HUBER: distance = |x1-x2| if |x1-x2| <= c else c * (|x1-x2| - c/2)
    """

    L1 = cv2.DIST_L1
    L2 = cv2.DIST_L2
    C = cv2.DIST_C
    L12 = cv2.DIST_L12
    FAIR = cv2.DIST_FAIR
    WELSCH = cv2.DIST_WELSCH
    HUBER = cv2.DIST_HUBER


@fn.NodeDecorator(
    node_id="cv2.distance_transform",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
    ],
    default_render_options={"data": {"src": "out"}},
)
def distance_transform(
    img: ImageFormat,
    distance_type: DistanceTypes = DistanceTypes.L1,
    mask_size: Literal[0, 3, 5] = 3,
) -> OpenCVImageFormat:
    distance_type = DistanceTypes.v(distance_type)
    img = cv2.cvtColor(assert_opencvdata(img), cv2.COLOR_BGR2GRAY)
    return NumpyImageFormat(
        cv2.distanceTransform(
            img,
            distance_type,
            int(mask_size),
        )
    )


@fn.NodeDecorator(
    node_id="cv2.watershed",
)
def watershed(
    img: ImageFormat,
    markers: Union[ImageFormat, np.ndarray],
) -> np.ndarray:
    markers: np.ndarray = assert_opencvdata(img, 1)
    img = assert_opencvdata(img)
    markers = markers.astype(np.int32)

    return cv2.watershed(img, markers)


NODE_SHELF = fn.Shelf(
    name="Masking and Thresholding",
    description="OpenCV image masking and thresholding nodes.",
    subshelves=[],
    nodes=[
        threshold,
        auto_threshold,
        adaptive_threshold,
        distance_transform,
        watershed,
    ],
)
