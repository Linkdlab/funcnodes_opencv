"""
filter implementations of openCV for funcnodes
"""

from typing import Optional, Tuple, Literal
import cv2
import numpy as np
from .imageformat import OpenCVImageFormat, ImageFormat
import funcnodes as fn
from .utils import assert_opencvdata


class BorderTypes(fn.DataEnum):
    """
    Border types for cv2.filter2D and cv2.blur

    Attributes:
        CONSTANT: cv2.BORDER_CONSTANT: Border is filled with the constant value
        REPLICATE: cv2.BORDER_REPLICATE: Border is replicated from the edge pixels
        REFLECT: cv2.BORDER_REFLECT: Border is reflectively mirrored
        REFLECT_101: cv2.BORDER_REFLECT_101: Border is reflectively mirrored with the edge pixels excluded
        WRAP: cv2.BORDER_WRAP: Border is wrapped around
    """

    CONSTANT = cv2.BORDER_CONSTANT
    REPLICATE = cv2.BORDER_REPLICATE
    REFLECT = cv2.BORDER_REFLECT
    REFLECT_101 = cv2.BORDER_REFLECT_101
    WRAP = cv2.BORDER_WRAP


@fn.NodeDecorator(
    node_id="cv2.filter2D",
    default_render_options={"data": {"src": "out"}},
)
def filter2D(
    img: ImageFormat,
    kernel: Optional[np.ndarray],
    ddepth: int = -1,
    anchor: Optional[Tuple[int, int]] = None,
    delta: int = 0,
    borderType: int = cv2.BORDER_DEFAULT,
) -> OpenCVImageFormat:
    if anchor is None:
        anchor = (-1, -1)

    data = assert_opencvdata(img)

    return OpenCVImageFormat(
        cv2.filter2D(
            data,
            ddepth=ddepth,
            kernel=kernel,
            anchor=anchor,
            delta=delta,
            borderType=borderType,
        )
    )


@fn.NodeDecorator(
    node_id="cv2.bilateralFilter",
    default_render_options={"data": {"src": "out"}},
)
def bilateralFilter(
    img: ImageFormat,
    d: int = 9,
    sigmaColor: float = 75,
    sigmaSpace: float = 75,
) -> OpenCVImageFormat:
    return OpenCVImageFormat(
        cv2.bilateralFilter(assert_opencvdata(img), d, sigmaColor, sigmaSpace)
    )


@fn.NodeDecorator(
    node_id="cv2.blur",
    default_render_options={"data": {"src": "out"}},
)
def blur(
    img: ImageFormat,
    kw: int = 5,
    kh: int = 0,
) -> OpenCVImageFormat:
    if kh <= 0:
        kh = kw

    ksize = (kw, kh)
    return OpenCVImageFormat(cv2.blur(assert_opencvdata(img), ksize))


@fn.NodeDecorator(
    node_id="cv2.boxFilter",
    default_render_options={"data": {"src": "out"}},
)
def boxFilter(
    img: ImageFormat,
    kw: int = 5,
    kh: Optional[int] = None,
    normalize: bool = True,
) -> OpenCVImageFormat:
    if kh is None:
        kh = kw

    if kw % 2 == 0:
        kw += 1

    if kh % 2 == 0:
        kh += 1

    ksize = (kw, kh)
    return OpenCVImageFormat(
        cv2.boxFilter(assert_opencvdata(img), -1, ksize=ksize, normalize=normalize)
    )


@fn.NodeDecorator(
    node_id="cv2.dilate",
    default_render_options={"data": {"src": "out"}},
)
def dilate(
    img: ImageFormat,
    kernel: Optional[np.ndarray] = None,
    iterations: int = 1,
) -> OpenCVImageFormat:
    return OpenCVImageFormat(
        cv2.dilate(assert_opencvdata(img), kernel=kernel, iterations=iterations)
    )


@fn.NodeDecorator(
    node_id="cv2.erode",
    default_render_options={"data": {"src": "out"}},
)
def erode(
    img: ImageFormat,
    kernel: Optional[np.ndarray] = None,
    iterations: int = 1,
) -> OpenCVImageFormat:
    return OpenCVImageFormat(
        cv2.erode(assert_opencvdata(img), kernel=kernel, iterations=iterations)
    )


@fn.NodeDecorator(
    node_id="cv2.GaussianBlur",
    default_render_options={"data": {"src": "out"}},
)
def GaussianBlur(
    img: ImageFormat,
    kw: int = 5,
    kh: Optional[int] = None,
    sigmaX: float = 1,
    sigmaY: Optional[float] = 0,
) -> OpenCVImageFormat:
    if kh is None:
        kh = kw

    if kw % 2 == 0:
        kw += 1

    if kh % 2 == 0:
        kh += 1
    ksize = (kw, kh)
    if sigmaY is None:
        sigmaY = sigmaX
    return OpenCVImageFormat(
        cv2.GaussianBlur(assert_opencvdata(img), ksize, sigmaX, sigmaY)
    )


@fn.NodeDecorator(
    node_id="cv2.Laplacian",
    default_render_options={"data": {"src": "out"}},
)
def Laplacian(
    img: ImageFormat,
    ksize: int = 1,
    scale: float = 1,
    delta: int = 0,
) -> OpenCVImageFormat:
    if ksize % 2 == 0:
        ksize += 1
    return OpenCVImageFormat(
        cv2.Laplacian(assert_opencvdata(img), -1, ksize=ksize, scale=scale, delta=delta)
    )


@fn.NodeDecorator(
    node_id="cv2.medianBlur",
    default_render_options={"data": {"src": "out"}},
)
def medianBlur(
    img: ImageFormat,
    ksize: int = 5,
) -> OpenCVImageFormat:
    if ksize % 2 == 0:
        ksize += 1
    return OpenCVImageFormat(cv2.medianBlur(assert_opencvdata(img), ksize))


@fn.NodeDecorator(
    node_id="cv2.pyrDown",
    default_render_options={"data": {"src": "out"}},
)
def pyrDown(
    img: ImageFormat,
) -> OpenCVImageFormat:
    return OpenCVImageFormat(cv2.pyrDown(assert_opencvdata(img)))


@fn.NodeDecorator(
    node_id="cv2.pyrUp",
    default_render_options={"data": {"src": "out"}},
)
def pyrUp(
    img: ImageFormat,
) -> OpenCVImageFormat:
    return OpenCVImageFormat(cv2.pyrUp(assert_opencvdata(img)))


@fn.NodeDecorator(
    node_id="cv2.Scharr",
    default_render_options={"data": {"src": "out"}},
)
def Scharr(
    img: ImageFormat,
    dx: int = 1,
    dy: int = 0,
    scale: float = 1,
    delta: int = 0,
) -> OpenCVImageFormat:
    return OpenCVImageFormat(
        cv2.Scharr(assert_opencvdata(img), -1, dx=dx, dy=dy, scale=scale, delta=delta)
    )


@fn.NodeDecorator(
    node_id="cv2.Sobel",
    default_render_options={"data": {"src": "out"}},
)
def Sobel(
    img: ImageFormat,
    dx: int = 1,
    dy: int = 0,
    ksize: Literal[1, 3, 5, 7] = 3,
    scale: float = 1,
    delta: int = 0,
) -> OpenCVImageFormat:
    return OpenCVImageFormat(
        cv2.Sobel(
            assert_opencvdata(img),
            -1,
            dx=dx,
            dy=dy,
            ksize=int(ksize),
            scale=scale,
            delta=delta,
        )
    )


@fn.NodeDecorator(
    node_id="cv2.stackBlur",
    default_render_options={"data": {"src": "out"}},
)
def stackBlur(
    img: ImageFormat,
    kw: int = 5,
    kh: Optional[int] = None,
) -> OpenCVImageFormat:
    if kh is None:
        kh = kw
    if kw % 2 == 0:
        kw += 1
    if kh % 2 == 0:
        kh += 1
    ksize = (kw, kh)
    return OpenCVImageFormat(cv2.stackBlur(assert_opencvdata(img), ksize))


NODE_SHELF = fn.Shelf(
    name="Filter",
    nodes=[
        filter2D,
        bilateralFilter,
        blur,
        boxFilter,
        dilate,
        erode,
        GaussianBlur,
        Laplacian,
        medianBlur,
        pyrDown,
        pyrUp,
        Scharr,
        Sobel,
        stackBlur,
    ],
    description="Nodes for image filtering",
    subshelves=[],
)

#
