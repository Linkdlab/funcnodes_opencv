from typing import Union, Optional, Tuple, List
import cv2
import numpy as np
import funcnodes as fn

from .imageformat import OpenCVImageFormat, ImageFormat
from .utils import assert_opencvdata


def rgb_from_hexstring(hexstring: str) -> Tuple[int, int, int]:
    if hexstring.startswith("#"):
        hexstring = hexstring[1:]
    return tuple(int(hexstring[i : i + 2], 16) for i in (0, 2, 4))


class LineTypes(fn.DataEnum):
    LINE_4 = cv2.LINE_4
    LINE_8 = cv2.LINE_8
    LINE_AA = cv2.LINE_AA
    FILLED = cv2.FILLED


@fn.NodeDecorator(
    "cv2.circle",
    name="circle",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Draws a circle on an image.",
)
@fn.controlled_wrapper(cv2.circle, wrapper_attribute="__fnwrapped__")
def circle(
    img: ImageFormat,
    center_x: Union[float, List[float], int, List[int]],
    center_y: Union[float, List[float], int, List[int]] = None,
    radius: Union[float, List[float], int, List[int]] = None,
    color: Optional[str] = "00FF00",
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
    shift: int = 0,
) -> OpenCVImageFormat:
    if isinstance(center_x, (float, int)):
        center_x = [center_x]
    if isinstance(center_y, (float, int)):
        center_y = [center_y]
    if isinstance(radius, (float, int)):
        radius = [radius]

    center_x = np.array(center_x)
    if center_y is None:
        # this means x, y, r are passed as a single list
        center_x = np.atleast_2d(center_x).T

        if radius is None:
            if center_x.shape[1] != 3 and center_x.shape[0] == 3:
                center_x = center_x.T
            if center_x.shape[1] != 3:
                raise ValueError(
                    "if center_y and radius is not provided, center_x must be a list of 3 elements (x, y, r)"
                )

            center_y = center_x[:, 1]
            radius = center_x[:, 2]
            center_x = center_x[:, 0]
        else:
            if center_x.shape[1] != 2 and center_x.shape[0] == 2:
                center_x = center_x.T
            if center_x.shape[1] != 2:
                raise ValueError(
                    "if center_y is not provided, center_x must be a list of at least 2 elements (x, y)"
                )

            center_y = center_x[:, 1]
            center_x = center_x[:, 0]

    center_y = np.array(center_y)
    radius = np.array(radius)

    assert len(center_x) == len(center_y) == len(radius), (
        "center_x, center_y, and radius lists must have the same length"
    )
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)
    img = assert_opencvdata(img, 3)

    for i in range(len(center_x)):
        cent = (int(center_x[i]), int(center_y[i]))
        rad = int(radius[i])
        img = cv2.circle(
            img=img,
            center=cent,
            radius=rad,
            color=color,
            thickness=thickness,
            lineType=lineType,
            shift=int(shift),
        )
    return OpenCVImageFormat(img)


@fn.NodeDecorator(
    "cv2.ellipse",
    name="ellipse",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Draws an ellipse on an image.",
)
@fn.controlled_wrapper(cv2.ellipse, wrapper_attribute="__fnwrapped__")
def ellipse(
    img: ImageFormat,
    center_x: Union[float, List[float], int, List[int]],
    center_y: Union[float, List[float], int, List[int]],
    axes_x: Union[float, List[float], int, List[int]],
    axes_y: Union[float, List[float], int, List[int]],
    angle: Union[float, List[float], int, List[int]],
    start_angle: int = 0,
    end_angle: int = 360,
    color: Optional[str] = "00FF00",
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
    shift: int = 0,
) -> OpenCVImageFormat:
    if isinstance(center_x, (float, int)):
        center_x = [center_x]
    if isinstance(center_y, (float, int)):
        center_y = [center_y]
    if isinstance(axes_x, (float, int)):
        axes_x = [axes_x]
    if isinstance(axes_y, (float, int)):
        axes_y = [axes_y]
    if isinstance(angle, (float, int)):
        angle = [angle]

    assert len(center_x) == len(center_y) == len(axes_x) == len(axes_y) == len(angle), (
        "center_x, center_y, axes_x, axes_y, and angle lists must have the same length"
    )
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)
    img = assert_opencvdata(img, 3)
    for i in range(len(center_x)):
        center = (int(center_x[i]), int(center_y[i]))
        axes = (int(axes_x[i]), int(axes_y[i]))
        ang = int(angle[i])
        img = cv2.ellipse(
            img=img,
            center=center,
            axes=axes,
            angle=ang,
            startAngle=start_angle,
            endAngle=end_angle,
            color=color,
            thickness=thickness,
            lineType=lineType,
            shift=int(shift),
        )
    return OpenCVImageFormat(img)


@fn.NodeDecorator(
    node_id="cv2.line",
    name="line",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Draws a line on an image.",
)
@fn.controlled_wrapper(cv2.line, wrapper_attribute="__fnwrapped__")
def line(
    img: ImageFormat,
    start_x: Union[float, List[float], int, List[int]],
    start_y: Union[float, List[float], int, List[int]] = None,
    end_x: Union[float, List[float], int, List[int]] = None,
    end_y: Union[float, List[float], int, List[int]] = None,
    color: Optional[str] = "00FF00",
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
    shift: int = 0,
) -> OpenCVImageFormat:
    if isinstance(start_x, (float, int)):
        start_x = [start_x]
    if isinstance(start_y, (float, int)):
        start_y = [start_y]
    if isinstance(end_x, (float, int)):
        end_x = [end_x]
    if isinstance(end_y, (float, int)):
        end_y = [end_y]

    start_x = np.array(start_x)

    if end_x is None:
        # this means x1, y1, x2, y2 are passed as a single list
        start_x = np.atleast_2d(start_x).T
        if start_x.shape[1] != 4 and start_x.shape[0] == 4:
            start_x = start_x.T
        if start_x.shape[1] != 4:
            raise ValueError(
                f"if end_x is not provided, start_x must be a list of 4 elements (x1, y1, x2, y2), got {start_x.shape}"
            )

        start_y = start_x[:, 1]
        end_x = start_x[:, 2]
        end_y = start_x[:, 3]
        start_x = start_x[:, 0]

    else:
        # end_x is provided
        if start_y is None and end_y is None:
            # start_x and end_x are (x1,y1), (x2, y2) pairs
            start_x = np.atleast_2d(start_x).T
            if start_x.shape[1] != 2 and start_x.shape[0] == 2:
                start_x = start_x.T
            if end_x.shape[1] != 2 and end_x.shape[0] == 2:
                end_x = end_x.T

            if start_x.shape[1] != 2 or end_x.shape[1] != 2:
                raise ValueError(
                    "if end_x is provided and ys are not, start_x and end-x must be a list of 2 elements (x, y). "
                    f"got {start_x.shape}, {end_x.shape}"
                )
            start_y = start_x[:, 1]
            start_x = start_x[:, 0]
            end_y = end_x[:, 1]
            end_x = end_x[:, 0]

    if start_y is None or end_y is None or end_x is None:
        raise ValueError(
            "error parsing inputs, valid combinations are: \n"
            "start_x:[(x1,y1,x2,y2)]\n"
            "start_x:[(x1,y1)], end_x:[(x2,y2)]\n"
            "start_x:[x1], start_y:[y1], end_x:[x2], end_y:[y2]"
        )

    start_y = np.array(start_y)
    end_x = np.array(end_x)
    end_y = np.array(end_y)
    assert len(start_x) == len(start_y) == len(end_x) == len(end_y), (
        "start_x, start_y, end_x, and end_y lists must have the same length"
    )
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)
    img = assert_opencvdata(img, 3)
    for i in range(len(start_x)):
        start = (int(start_x[i]), int(start_y[i]))
        end = (int(end_x[i]), int(end_y[i]))
        img = cv2.line(
            img=img,
            pt1=start,
            pt2=end,
            color=color,
            thickness=thickness,
            lineType=lineType,
            shift=int(shift),
        )
    return OpenCVImageFormat(img)


@fn.NodeDecorator(
    node_id="cv2.rectangle",
    name="rectangle",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Draws a rectangle on an image.",
)
@fn.controlled_wrapper(cv2.rectangle, wrapper_attribute="__fnwrapped__")
def rectangle(
    img: ImageFormat,
    x: Union[float, List[float], int, List[int]],
    y: Union[float, List[float], int, List[int]],
    width: Union[float, List[float], int, List[int]],
    height: Union[float, List[float], int, List[int]],
    color: Optional[str] = "00FF00",
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
    shift: int = 0,
) -> OpenCVImageFormat:
    if isinstance(x, (float, int)):
        x = [x]
    if isinstance(y, (float, int)):
        y = [y]
    if isinstance(width, (float, int)):
        width = [width]
    if isinstance(height, (float, int)):
        height = [height]
    assert len(x) == len(y) == len(width) == len(height), (
        "x, y, width, and height lists must have the same length"
    )
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)
    img = assert_opencvdata(img, 3)
    for i in range(len(x)):
        start = (int(x[i]), int(y[i]))
        end = (int(x[i] + width[i]), int(y[i] + height[i]))
        img = cv2.rectangle(
            img=img,
            pt1=start,
            pt2=end,
            color=color,
            thickness=thickness,
            lineType=lineType,
            shift=int(shift),
        )
    return OpenCVImageFormat(img)


@fn.NodeDecorator(
    node_id="cv2.polylines",
    name="polylines",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Draws a polyline on an image.",
)
@fn.controlled_wrapper(cv2.polylines, wrapper_attribute="__fnwrapped__")
def polylines(
    img: ImageFormat,
    pts: List[np.ndarray],
    isClosed: bool = True,
    color: Optional[str] = "00FF00",
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
    shift: int = 0,
) -> OpenCVImageFormat:
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)

    if isinstance(pts, np.ndarray):
        if pts.ndim == 2:
            pts = [pts]

    img = cv2.polylines(
        img=assert_opencvdata(img, 3),
        pts=pts,
        isClosed=isClosed,
        color=color,
        thickness=thickness,
        lineType=lineType,
        shift=int(shift),
    )
    return OpenCVImageFormat(img)


@fn.NodeDecorator(
    node_id="cv2.fillPoly",
    name="fillPoly",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Fills a polygon on an image.",
)
@fn.controlled_wrapper(cv2.fillPoly, wrapper_attribute="__fnwrapped__")
def fillPoly(
    img: ImageFormat,
    pts: List[np.ndarray],
    color: Optional[str] = "00FF00",
    lineType: LineTypes = LineTypes.LINE_8,
    shift: int = 0,
) -> OpenCVImageFormat:
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)

    if isinstance(pts, np.ndarray):
        if pts.ndim == 2:
            pts = [pts]

    img = cv2.fillPoly(
        img=assert_opencvdata(img, 3),
        pts=pts,
        color=color,
        lineType=lineType,
        shift=int(shift),
    )
    return OpenCVImageFormat(img)


class FontTypes(fn.DataEnum):
    """
    Enum for font types.

    """

    FONT_HERSHEY_SIMPLEX = cv2.FONT_HERSHEY_SIMPLEX
    FONT_HERSHEY_PLAIN = cv2.FONT_HERSHEY_PLAIN
    FONT_HERSHEY_DUPLEX = cv2.FONT_HERSHEY_DUPLEX
    FONT_HERSHEY_COMPLEX = cv2.FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_TRIPLEX = cv2.FONT_HERSHEY_TRIPLEX
    FONT_HERSHEY_COMPLEX_SMALL = cv2.FONT_HERSHEY_COMPLEX_SMALL
    FONT_HERSHEY_SCRIPT_SIMPLEX = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_HERSHEY_SCRIPT_COMPLEX = cv2.FONT_HERSHEY_SCRIPT_COMPLEX


@fn.NodeDecorator(
    node_id="cv2.putText",
    name="putText",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Draws text on an image.",
)
@fn.controlled_wrapper(cv2.putText, wrapper_attribute="__fnwrapped__")
def putText(
    img: ImageFormat,
    text: str,
    org_x: Union[float, List[float], int, List[int]],
    org_y: Union[float, List[float], int, List[int]],
    fontFace: FontTypes = FontTypes.FONT_HERSHEY_SIMPLEX,
    fontScale: float = 1,
    color: Optional[str] = "00FF00",
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
    bottomLeftOrigin: bool = False,
) -> OpenCVImageFormat:
    if isinstance(org_x, (float, int)):
        org_x = [org_x]
    if isinstance(org_y, (float, int)):
        org_y = [org_y]
    assert len(org_x) == len(org_y), "org_x and org_y lists must have the same length"
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)
    img = assert_opencvdata(img, 3)
    for i in range(len(org_x)):
        org = (int(org_x[i]), int(org_y[i]))
        img = cv2.putText(
            img=img,
            text=text,
            org=org,
            fontFace=FontTypes.v(fontFace),
            fontScale=fontScale,
            color=color,
            thickness=thickness,
            lineType=lineType,
            bottomLeftOrigin=bottomLeftOrigin,
        )
    return OpenCVImageFormat(img)


@fn.NodeDecorator(
    node_id="cv2.arrowedLine",
    name="arrowedLine",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Draws an arrowed line on an image.",
)
@fn.controlled_wrapper(cv2.arrowedLine, wrapper_attribute="__fnwrapped__")
def arrowedLine(
    img: ImageFormat,
    start_x: Union[float, List[float], int, List[int]],
    start_y: Union[float, List[float], int, List[int]],
    end_x: Union[float, List[float], int, List[int]],
    end_y: Union[float, List[float], int, List[int]],
    color: Optional[str] = "00FF00",
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
    shift: int = 0,
    tipLength: float = 0.1,
) -> OpenCVImageFormat:
    if isinstance(start_x, (float, int)):
        start_x = [start_x]
    if isinstance(start_y, (float, int)):
        start_y = [start_y]
    if isinstance(end_x, (float, int)):
        end_x = [end_x]
    if isinstance(end_y, (float, int)):
        end_y = [end_y]
    assert len(start_x) == len(start_y) == len(end_x) == len(end_y), (
        "start_x, start_y, end_x, and end_y lists must have the same length"
    )
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0
    lineType = LineTypes.v(lineType)
    img = assert_opencvdata(img, 3)
    for i in range(len(start_x)):
        start = (int(start_x[i]), int(start_y[i]))
        end = (int(end_x[i]), int(end_y[i]))
        img = cv2.arrowedLine(
            img=img,
            pt1=start,
            pt2=end,
            color=color,
            thickness=thickness,
            line_type=lineType,
            shift=int(shift),
            tipLength=tipLength,
        )
    return OpenCVImageFormat(img)


@fn.NodeDecorator(
    node_id="cv2.fillConvexPoly",
    name="fillConvexPoly",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Fills a convex polygon on an image.",
)
@fn.controlled_wrapper(cv2.fillConvexPoly, wrapper_attribute="__fnwrapped__")
def fillConvexPoly(
    img: ImageFormat,
    pts: List[np.ndarray],
    color: Optional[str] = "00FF00",
    lineType: LineTypes = LineTypes.LINE_8,
    shift: int = 0,
) -> OpenCVImageFormat:
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)

    img = cv2.fillConvexPoly(
        img=assert_opencvdata(img, 3),
        points=pts,
        color=color,
        lineType=lineType,
        shift=int(shift),
    )
    return OpenCVImageFormat(img)


class MarkerTypes(fn.DataEnum):
    """
    Enum for marker types.

    """

    CROSS = cv2.MARKER_CROSS
    TILTED_CROSS = cv2.MARKER_TILTED_CROSS
    STAR = cv2.MARKER_STAR
    DIAMOND = cv2.MARKER_DIAMOND
    SQUARE = cv2.MARKER_SQUARE
    TRIANGLE_UP = cv2.MARKER_TRIANGLE_UP
    TRIANGLE_DOWN = cv2.MARKER_TRIANGLE_DOWN


@fn.NodeDecorator(
    "cv2.drawMarker",
    name="drawMarker",
    default_render_options={
        "io": {
            "color": {"type": "color"},
        },
        "data": {"src": "out"},
    },
    description="Draws a marker on an image.",
)
@fn.controlled_wrapper(cv2.drawMarker, wrapper_attribute="__fnwrapped__")
def drawMarker(
    img: ImageFormat,
    pos_x: Union[float, List[float], int, List[int]],
    pos_y: Union[float, List[float], int, List[int]],
    color: Optional[str] = "00FF00",
    markerType: MarkerTypes = MarkerTypes.CROSS,
    markerSize: int = 20,
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
) -> OpenCVImageFormat:
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    lineType = LineTypes.v(lineType)
    markerType = MarkerTypes.v(markerType)

    if isinstance(pos_x, (float, int)):
        pos_x = [pos_x]
    if isinstance(pos_y, (float, int)):
        pos_y = [pos_y]
    assert len(pos_x) == len(pos_y), "pos_x and pos_y lists must have the same length"

    img = assert_opencvdata(img, 3)
    for i in range(len(pos_x)):
        position = (int(pos_x[i]), int(pos_y[i]))
        img = cv2.drawMarker(
            img=img,
            position=position,
            color=color,
            markerType=markerType,
            markerSize=int(markerSize),
            thickness=thickness,
            line_type=lineType,
        )
    return OpenCVImageFormat(img)


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
@fn.controlled_wrapper(cv2.drawContours, wrapper_attribute="__fnwrapped__")
def drawContours(
    img: ImageFormat,
    contours: np.ndarray,
    contourIdx: int = -1,
    color: Optional[str] = "00FF00",
    thickness: int = 1,
    lineType: LineTypes = LineTypes.LINE_8,
    offset_dx: int = 0,
    offset_dy: int = 0,
) -> OpenCVImageFormat:
    color = np.array(rgb_from_hexstring(color))[::-1] / 255.0

    offset = (offset_dx, offset_dy)
    lineType = LineTypes.v(lineType)

    return OpenCVImageFormat(
        cv2.drawContours(
            image=assert_opencvdata(img, 3),
            contours=contours,
            contourIdx=contourIdx,
            color=color,
            thickness=thickness,
            lineType=lineType,
            offset=offset,
        )
    )


class ColorMap(fn.DataEnum):
    """
    Enum for color maps.

    """

    AUTUMN = cv2.COLORMAP_AUTUMN
    BONE = cv2.COLORMAP_BONE
    JET = cv2.COLORMAP_JET
    WINTER = cv2.COLORMAP_WINTER
    RAINBOW = cv2.COLORMAP_RAINBOW
    OCEAN = cv2.COLORMAP_OCEAN
    SUMMER = cv2.COLORMAP_SUMMER
    SPRING = cv2.COLORMAP_SPRING
    COOL = cv2.COLORMAP_COOL
    HSV = cv2.COLORMAP_HSV
    PINK = cv2.COLORMAP_PINK
    HOT = cv2.COLORMAP_HOT
    PARULA = cv2.COLORMAP_PARULA
    MAGMA = cv2.COLORMAP_MAGMA
    INFERNO = cv2.COLORMAP_INFERNO
    PLASMA = cv2.COLORMAP_PLASMA
    VIRIDIS = cv2.COLORMAP_VIRIDIS
    CIVIDIS = cv2.COLORMAP_CIVIDIS
    TWILIGHT = cv2.COLORMAP_TWILIGHT
    TWILIGHT_SHIFTED = cv2.COLORMAP_TWILIGHT_SHIFTED
    TURBO = cv2.COLORMAP_TURBO
    DEEPGREEN = cv2.COLORMAP_DEEPGREEN


def extract_boundaries(label_img):
    boundaries = {}
    labels = np.unique(label_img)
    for label in labels:
        mask = (label_img == label).astype(np.uint8)
        edges = cv2.Canny(mask, 0, 1)  # Fast edge detection
        points = np.column_stack(np.where(edges > 0))
        if points.size == 0:
            points = np.column_stack(np.where(mask > 0))  # fallback
        # get a subset of points (most northher, nothwest, etc.)

        boundaries[label] = points
    return boundaries


def optimize_label_distribution(label_img: np.ndarray) -> np.ndarray:
    """
    Reorders labels in a label image to optimize their spatial distribution.

    The function treats negative labels as fixed while positive labels are reordered
    based on their spatial centroids. For each positive label, it computes a centroid
    (using contours when available, or falling back to the mean of pixel coordinates).
    Then, it computes a pairwise distance matrix between centroids and derives a force
    matrix based on an inverse square law. A greedy algorithm is applied to select a new
    ordering for the positive labels that aims to spread them out evenly.

    Parameters
    ----------
    label_img : np.ndarray
        A 2D numpy array of integer type where each unique integer represents a distinct label.
        Negative values are preserved and only positive labels are reordered.

    Returns
    -------
    np.ndarray
        A new label image with re-mapped labels. Negative labels remain unchanged while positive
        labels are reordered based on the optimized distribution.

    Raises
    ------
    TypeError
        If `label_img` is not a numpy array.
    ValueError
        If `label_img` is not of an integer type.
    RuntimeError
        If an unexpected number of return values is received from cv2.findContours.
    """
    # Validate input
    if not isinstance(label_img, np.ndarray):
        raise TypeError("label_img must be a numpy array.")
    if not np.issubdtype(label_img.dtype, np.integer):
        raise ValueError("label_img must be of an integer type.")

    # Step 0: Identify unique labels and separate negative and positive labels
    unique_labels = np.unique(label_img)
    negative_mask = unique_labels < 0
    negative_labels = unique_labels[negative_mask]
    positive_labels = unique_labels[~negative_mask]

    # Step 1: Compute centroids for positive labels
    centroids = {}
    for label in positive_labels:
        # Create a binary mask for the current label
        mask = (label_img == label).astype(np.uint8)

        # Find contours; handle differences between OpenCV versions
        contours_result = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if len(contours_result) == 2:
            contours, _ = contours_result
        elif len(contours_result) == 3:
            # OpenCV 3 returns (image, contours, hierarchy)
            _, contours, _ = contours_result
        else:
            raise RuntimeError(
                "Unexpected number of return values from cv2.findContours"
            )

        # Try to compute centroid using moments from the first contour (if valid)
        if contours and len(contours[0]) >= 3:
            M = cv2.moments(contours[0])
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]
                centroids[label] = np.array([cy, cx], dtype=np.float32)
                continue

        # Fallback: compute centroid from the pixel indices directly
        ys, xs = np.nonzero(mask)
        if xs.size > 0:
            centroids[label] = np.array([np.mean(ys), np.mean(xs)], dtype=np.float32)
        else:
            # Default to the image center if no pixels are found (should not happen)
            centroids[label] = np.array(
                [label_img.shape[0] / 2, label_img.shape[1] / 2], dtype=np.float32
            )

    # Step 2: Compute pairwise distance matrix for positive labels
    n_pos = len(positive_labels)
    if n_pos == 0:
        # If there are no positive labels, return a copy of the input image
        return label_img.copy()

    distance_matrix = np.zeros((n_pos, n_pos), dtype=np.float32)
    for i in range(n_pos):
        for j in range(i + 1, n_pos):
            pt1 = centroids[positive_labels[i]]
            pt2 = centroids[positive_labels[j]]
            dist = np.linalg.norm(pt1 - pt2)
            distance_matrix[i, j] = dist
            distance_matrix[j, i] = dist

    # Compute force matrix using an inverse square law (with a small epsilon to avoid division by zero)
    epsilon = 1e-6
    force_matrix = -1 / (distance_matrix + epsilon) ** 2

    # Step 3: Optimize ordering of positive labels using a greedy algorithm
    # Start with the label having the smallest average force (least repulsion)
    start_label_idx = np.nanmean(force_matrix, axis=1).argmin()
    label_order = [start_label_idx]
    used = {start_label_idx}

    while len(label_order) < n_pos:
        used_indices = np.array(list(used), dtype=np.int32)
        # Sum the forces from all used labels for each candidate label
        summed_forces = force_matrix[used_indices].sum(axis=0)
        # Exclude already used labels by marking their force as NaN
        summed_forces[used_indices] = np.nan
        next_idx = np.nanargmax(summed_forces)
        label_order.append(next_idx)
        used.add(next_idx)

    # Create new ordering for positive labels based on the computed order
    ordered_positive_labels = positive_labels[np.array(label_order)]

    # Step 4: Combine negative labels (unchanged) with the newly ordered positive labels
    final_label_list = np.concatenate([negative_labels, ordered_positive_labels])

    # Create a mapping from old labels to new labels.
    # This mapping uses the order of unique_labels and final_label_list so that negative labels remain unchanged.
    mapping = {old: new for old, new in zip(unique_labels, final_label_list)}

    # Remap the label image using the computed mapping
    remapped = np.vectorize(mapping.get)(label_img)

    return remapped


@fn.NodeDecorator(
    node_id="cv2.labels_to_color",
    default_render_options={"data": {"src": "out"}},
)
def labels_to_color(
    labels: np.ndarray, colormap: ColorMap = ColorMap.JET, mix: bool = True
) -> OpenCVImageFormat:
    """
    Takes an 2d array of labels and converts it to a color image.
    Apply a colormap to the labels.

    Args:
        labels: 2d array of labels.
        colormap: Colormap to use.
        mix: bool, if True, mix the numerical values of the labels to prevent a spread of colors.
    """

    labels = np.array(labels)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2D array")
    unique_labels = np.unique(labels)
    if mix:
        labels = optimize_label_distribution(labels)

    n_colors = min(len(unique_labels), 256)
    colors = cv2.applyColorMap(
        np.linspace(0, 255, n_colors).reshape(n_colors, 1).astype(np.uint8),
        ColorMap.v(colormap),
    )

    img = np.zeros(labels.shape + (3,)) * np.nan
    for i, label in enumerate(unique_labels):
        img[labels == label] = colors[i % n_colors]

    return OpenCVImageFormat(img.astype(np.uint8))


NODE_SHELF = fn.Shelf(
    nodes=[
        circle,
        ellipse,
        line,
        rectangle,
        polylines,
        fillPoly,
        putText,
        arrowedLine,
        drawContours,
        labels_to_color,
    ],
    subshelves=[],
    name="Drawing",
    description="Nodes for drawing shapes and text on images.",
)
