from typing import Optional, Tuple
import cv2
import numpy as np
from .imageformat import OpenCVImageFormat, ImageFormat
import funcnodes as fn
from exposedfunctionality import controlled_wrapper
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
)
@controlled_wrapper(cv2.findContours, wrapper_attribute="__fnwrapped__")
def _findContours(
    img: ImageFormat,
    mode: RetrievalModes = RetrievalModes.EXTERNAL,
    method: ContourApproximationModes = ContourApproximationModes.SIMPLE,
    offset_dx: int = 0,
    offset_dy: int = 0,
) -> list:
    offset = (offset_dx, offset_dy)
    mode = RetrievalModes.v(mode)
    method = ContourApproximationModes.v(method)

    contours, hierarchy = cv2.findContours(
        image=assert_opencvdata(img, 1),
        mode=mode,
        method=method,
        offset=offset,
    )

    return list(contours)


Structural_Analysis_and_Shape_Descriptors_NODE_SHELF = fn.Shelf(
    nodes=[_findContours],
    subshelves=[],
    name="Structural Analysis and Shape Descriptors",
    description="",
)


class LineTypes(fn.DataEnum):
    LINE_4 = cv2.LINE_4
    LINE_8 = cv2.LINE_8
    LINE_AA = cv2.LINE_AA
    FILLED = cv2.FILLED


def rgb_from_hexstring(hexstring: str) -> Tuple[int, int, int]:
    return tuple(int(hexstring[i : i + 2], 16) for i in (0, 2, 4))


@fn.NodeDecorator(
    "cv2.drawContours",
    name="drawContours",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
)
@controlled_wrapper(cv2.drawContours, wrapper_attribute="__fnwrapped__")
def _drawContours(
    img: ImageFormat,
    contours: np.ndarray,
    contourIdx: int = -1,
    color: Optional[str] = "00FF00",
    thickness=1,
    lineType: LineTypes = LineTypes.LINE_8,
    offset_dx: int = 0,
    offset_dy: int = 0,
) -> OpenCVImageFormat:
    color = rgb_from_hexstring(color)

    color = color[::-1]

    offset = (offset_dx, offset_dy)
    lineType = LineTypes.v(lineType)

    return OpenCVImageFormat(
        cv2.drawContours(
            image=assert_opencvdata(img),
            contours=contours,
            contourIdx=contourIdx,
            color=color,
            thickness=thickness,
            lineType=lineType,
            offset=offset,
        )
    )


Drawing_Functions_NODE_SHELF = fn.Shelf(
    nodes=[_drawContours],
    subshelves=[],
    name="Drawing_Functions",
    description="",
)


Image_Processing_NODE_SHELF = fn.Shelf(
    nodes=[],
    subshelves=[
        Drawing_Functions_NODE_SHELF,
        Structural_Analysis_and_Shape_Descriptors_NODE_SHELF,
    ],
    name="Image Processing",
    description="",
)
