"""
Components functions of opencv for funcnodes
"""

from typing import Literal, Tuple
import funcnodes as fn
import cv2
import numpy as np
from .imageformat import OpenCVImageFormat, ImageFormat
from .utils import normalize, LUT, assert_opencvdata


@fn.NodeDecorator(
    node_id="cv2.connectedComponents",
    outputs=[
        {"name": "retval", "type": int},
        {"name": "labels", "type": np.ndarray},
    ],
    default_render_options={"data": {"src": "out"}},
)
def connectedComponents(
    img: ImageFormat,
    connectivity: Literal[4, 8] = 8,
) -> Tuple[int, np.ndarray]:
    connectivity = int(connectivity)
    data = assert_opencvdata(img, 1)
    img = normalize(data)
    retval, labels = cv2.connectedComponents(img, connectivity=connectivity)
    return retval, labels


@fn.NodeDecorator(
    node_id="cv2.labels_to_color",
    default_render_options={"data": {"src": "out"}},
)
def labels_to_color(labels: np.ndarray) -> OpenCVImageFormat:
    """
    Convert labels to color image
    """
    labels = (labels % 256).astype(np.uint8)
    return OpenCVImageFormat(cv2.LUT(cv2.merge((labels, labels, labels)), LUT))


NODE_SHELF = fn.Shelf(
    name="Components",
    nodes=[
        connectedComponents,
        labels_to_color,
    ],
    description="Nodes for component analysis",
    subshelves=[],
)
