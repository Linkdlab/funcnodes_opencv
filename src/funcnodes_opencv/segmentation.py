# Segmentation & Contour Analysis

from typing import Literal, Tuple, List, Union
import funcnodes as fn
import cv2
import numpy as np
import pandas as pd
from .imageformat import ImageFormat, NumpyImageFormat
from .utils import assert_opencvdata


class RetrievalModes(fn.DataEnum):
    """
    Mode of the contour retrieval algorithm.

    Attributes:
        EXTERNAL: cv2.RETR_EXTERNAL: retrieves only the extreme outer contours.
            It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours and leaves them as leaves of the
            outer contour list. It sets hierarchy[i][2]=hierarchy[i][3]=-1 for all the contours.
        LIST: cv2.RETR_LIST: retrieves all of the contours without establishing any hierarchical relationships.
        CCOMP: cv2.RETR_CCOMP: retrieves all of the contours and organizes them into a two-level hierarchy.
        TREE: cv2.RETR_TREE: retrieves all of the contours and reconstructs a full hierarchy of nested contours.
        FLOODFILL: cv2.RETR_FLOODFILL
    """

    EXTERNAL = cv2.RETR_EXTERNAL
    LIST = cv2.RETR_LIST
    CCOMP = cv2.RETR_CCOMP
    TREE = cv2.RETR_TREE
    FLOODFILL = cv2.RETR_FLOODFILL


class ContourApproximationModes(fn.DataEnum):
    """
    Approximation modes for the contour retrieval algorithm.

    Attributes:
        NONE: cv2.CHAIN_APPROX_NONE: stores absolutely all the contour points.
        SIMPLE: cv2.CHAIN_APPROX_SIMPLE: compresses horizontal, vertical, and diagonal segments
        TC89_L1: cv2.CHAIN_APPROX_TC89_L1: applies one of the flavors of the Teh-Chin chain approximation algorithm
        TC89_KCOS: cv2.CHAIN_APPROX_TC89_KCOS: applies one of the flavors of the Teh-Chin chain approximation algorithm
    """

    NONE = cv2.CHAIN_APPROX_NONE
    SIMPLE = cv2.CHAIN_APPROX_SIMPLE
    TC89_L1 = cv2.CHAIN_APPROX_TC89_L1
    TC89_KCOS = cv2.CHAIN_APPROX_TC89_KCOS


@fn.NodeDecorator(
    "cv2.findContours",
    name="findContours",
    outputs=[
        {"name": "contours"},
    ],
    description="Finds contours in a binary image.",
)
@fn.controlled_wrapper(cv2.findContours, wrapper_attribute="__fnwrapped__")
def findContours(
    img: ImageFormat,
    mode: RetrievalModes = RetrievalModes.EXTERNAL,
    method: ContourApproximationModes = ContourApproximationModes.SIMPLE,
    offset_dx: int = 0,
    offset_dy: int = 0,
) -> List[np.ndarray]:
    offset = (offset_dx, offset_dy)

    mode = RetrievalModes.v(mode)
    method = ContourApproximationModes.v(method)

    img = assert_opencvdata(img, 1)
    if mode != RetrievalModes.FLOODFILL.value or mode != RetrievalModes.CCOMP.value:
        img = (img * 255).astype(np.uint8)

    if mode == RetrievalModes.FLOODFILL.value:
        img = (img * 255).astype(np.int32)  # cv2.FLOODFILL 32bit signed int

    contours, hierarchy = cv2.findContours(
        image=img,
        mode=mode,
        method=method,
        offset=offset,
    )

    return list(contours)


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
    default_render_options={"data": {"src": "out"}},
    description="Calculates the distance to the closest zero pixel for each pixel of the source image.",
)
def distance_transform(
    img: ImageFormat,
    distance_type: DistanceTypes = DistanceTypes.L1,
    mask_size: Literal[0, 3, 5] = 3,
) -> NumpyImageFormat:
    return NumpyImageFormat(
        cv2.distanceTransform(
            (assert_opencvdata(img, channel=1) * 255).astype(np.uint8),
            DistanceTypes.v(distance_type),
            int(mask_size),
        )
    )


class ConnectedComponentsAlgorithmsTypes(fn.DataEnum):
    """
    Connected components algorithms.

    Attributes:
        DEFAULT: cv2.CCL_DEFAULT: default algorithm.
        WU: cv2.CCL_WU: Wu's algorithm.
        GRANA: cv2.CCL_GRANA: Grana's algorithm.
        BOLELLI: cv2.CCL_BOLELLI: Bolelli's algorithm.
        SAUF: cv2.CCL_SAUF: SAUF algorithm.
        BBDT: cv2.CCL_BBDT: BBDT algorithm.
        SPAGHETTI: cv2.CCL_SPAGHETTI: Spaghetti algorithm.

    """

    DEFAULT = cv2.CCL_DEFAULT
    WU = cv2.CCL_WU
    GRANA = cv2.CCL_GRANA
    BOLELLI = cv2.CCL_BOLELLI
    SAUF = cv2.CCL_SAUF
    BBDT = cv2.CCL_BBDT
    SPAGHETTI = cv2.CCL_SPAGHETTI


@fn.NodeDecorator(
    node_id="cv2.connectedComponents",
    outputs=[
        {
            "name": "retval",
        },
        {
            "name": "labels",
        },
        {
            "name": "stats",
        },
    ],
    description="Finds connected components in a binary image.",
)
def connectedComponents(
    img: ImageFormat,
    connectivity: Literal[4, 8] = 8,
    algorithm: ConnectedComponentsAlgorithmsTypes = ConnectedComponentsAlgorithmsTypes.DEFAULT,
    background: Literal[-1, 0, 1] = 0,
) -> Tuple[int, np.ndarray, pd.DataFrame]:
    connectivity = int(connectivity)
    data = (assert_opencvdata(img, 1) * 255).astype(np.uint8)
    algorithm = ConnectedComponentsAlgorithmsTypes.v(algorithm)

    retval, labels, stats, centroids = cv2.connectedComponentsWithStatsWithAlgorithm(
        data,
        connectivity=connectivity,
        ltype=cv2.CV_32S,
        ccltype=algorithm,
    )

    indices = np.arange(1, retval + 1, dtype=np.int32)
    labels = labels.astype(np.int32)
    background = int(background)

    if background > 0:
        labels = labels + background
        indices = indices + background
    elif background < 0:
        labels[labels == 0] = background

    stats = stats[
        :,
        [
            cv2.CC_STAT_LEFT,
            cv2.CC_STAT_TOP,
            cv2.CC_STAT_WIDTH,
            cv2.CC_STAT_HEIGHT,
            cv2.CC_STAT_AREA,
        ],
    ]
    df_stats = pd.DataFrame(
        stats, columns=["left", "top", "width", "height", "area"], index=indices
    )
    df_stats["centroid_x"] = centroids[:, 0]
    df_stats["centroid_y"] = centroids[:, 1]

    return retval, NumpyImageFormat(labels), df_stats


@fn.NodeDecorator(
    node_id="cv2.watershed",
    description="Performs a marker-based image segmentation using the watershed algorithm.",
)
def watershed(
    img: ImageFormat,
    markers: Union[ImageFormat, np.ndarray],
) -> np.ndarray:
    if isinstance(markers, ImageFormat):
        markers = markers.data

    img = (assert_opencvdata(img, 3) * 255).astype(np.uint8)

    return cv2.watershed(img, markers)[:, :, 0]


NODE_SHELF = fn.Shelf(
    name="Segmentation & Contour Analysis",
    nodes=[findContours, distance_transform, connectedComponents, watershed],
    subshelves=[],
    description="Segmentation and contour analysis nodes.",
)
