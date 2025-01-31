import cv2
import numpy as np
from .imageformat import OpenCVImageFormat, ImageFormat
import funcnodes as fn
import math
from .utils import assert_opencvimg, assert_opencvdata


class Interpolations(fn.DataEnum):
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    AREA = cv2.INTER_AREA
    LANCZOS4 = cv2.INTER_LANCZOS4


class FlipCodes(fn.DataEnum):
    HORIZONTAL = 1
    VERTICAL = 0
    BOTH = -1


@fn.NodeDecorator(
    node_id="cv2.resize",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
    ],
    default_render_options={"data": {"src": "out"}},
)
def resize(
    img: ImageFormat,
    h: int = None,
    w: int = None,
    fh: float = 1,
    fw: float = 1,
    interpolation: Interpolations = Interpolations.LINEAR,
) -> OpenCVImageFormat:
    interpolation: int = Interpolations.v(interpolation)
    img = assert_opencvimg(img)
    if h is None:
        h = int(round(fh * img.height()))

    if w is None:
        w = int(round(fw * img.width()))

    return OpenCVImageFormat(
        cv2.resize(img.data, dsize=(w, h), interpolation=interpolation)
    )


@fn.NodeDecorator(
    node_id="cv2.flip",
    default_render_options={"data": {"src": "out"}},
)
def flip(
    img: ImageFormat,
    flip_code: FlipCodes = FlipCodes.HORIZONTAL,
) -> OpenCVImageFormat:
    flip_code = FlipCodes.v(flip_code)
    return OpenCVImageFormat(cv2.flip(assert_opencvdata(img), flip_code))


class RoationCode(fn.DataEnum):
    ROTATE_90_CLOCKWISE = cv2.ROTATE_90_CLOCKWISE
    ROTATE_180 = cv2.ROTATE_180
    ROTATE_90_COUNTERCLOCKWISE = cv2.ROTATE_90_COUNTERCLOCKWISE


@fn.NodeDecorator(
    node_id="cv2.rotate",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
    ],
    default_render_options={"data": {"src": "out"}},
)
def rotate(
    img: ImageFormat, rot: RoationCode = RoationCode.ROTATE_90_CLOCKWISE
) -> OpenCVImageFormat:
    rot = RoationCode.v(rot)
    return OpenCVImageFormat(cv2.rotate(assert_opencvdata(img), rot))


class FreeRotationCropMode(fn.DataEnum):
    """defines the mode of cropping for free rotation.
    KEEP: Rotates the images and fills the empty space with black.
    CROP: Rotates the images and crops the empty space.
    NONE: does not change the size of the image.
    """

    KEEP = 0
    CROP = 1
    NONE = 2


def rotatedRectWithMaxArea(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle (maximal area) within the rotated rectangle.
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2.0 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        #   the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


@fn.NodeDecorator(
    node_id="cv2.freeRotation",
    outputs=[
        {"name": "out", "type": OpenCVImageFormat},
        {"name": "M", "type": np.ndarray},
    ],
    default_render_options={"data": {"src": "out"}},
    default_io_options={"angle": {"value_options": {"min": 0, "max": 360}}},
)
def freeRotation(
    img: ImageFormat,
    angle: float = 0,
    mode: FreeRotationCropMode = FreeRotationCropMode.KEEP,
) -> OpenCVImageFormat:
    """
    Rotates the image by the given angle.
    Args:
        img: ImageFormat: The image to rotate.
        angle: float: The angle to rotate the image by.
        mode: FreeRotationCropMode: The mode of cropping for the rotation.
    Returns:
        funcnodes_opencv.OpenCVImageFormat: The rotated image.
    """

    img = assert_opencvdata(img)

    cx = img.shape[1] / 2
    cy = img.shape[0] / 2

    mode = FreeRotationCropMode.interfere(mode)
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    corners = np.array(
        [
            [0, 0],  # Top-left
            [img.shape[1], 0],  # Top-right
            [0, img.shape[0]],  # Bottom-left
            [img.shape[1], img.shape[0]],  # Bottom-right
        ],
        dtype=np.float32,
    )
    if mode == FreeRotationCropMode.NONE:
        # rotate the image by angle and keep the size of the image
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    elif mode == FreeRotationCropMode.KEEP:
        # rotate the image by angle and fill the empty space with black
        # calculate the new size of the image to fit the rotated image
        transformed_corners = cv2.transform(np.array([corners]), M)[0]
        min_x = min(transformed_corners[:, 0])
        max_x = max(transformed_corners[:, 0])
        min_y = min(transformed_corners[:, 1])
        max_y = max(transformed_corners[:, 1])

        w, h = int(max_x - min_x), int(max_y - min_y)

        # Adjust the rotation matrix to account for translation
        M[0, 2] += (w / 2) - cx
        M[1, 2] += (h / 2) - cy

        rotated = cv2.warpAffine(img, M, (w, h))
    elif mode == FreeRotationCropMode.CROP:
        # rotate the image by angle and crop the empty space
        transformed_corners = cv2.transform(np.array([corners]), M)[0]

        w, h = rotatedRectWithMaxArea(img.shape[1], img.shape[0], math.radians(angle))

        # Adjust the rotation matrix to account for translation
        M[0, 2] += (w / 2) - cx
        M[1, 2] += (h / 2) - cy

        rotated = cv2.warpAffine(img, M, (int(w), int(h)))

    return OpenCVImageFormat(rotated), M


@fn.NodeDecorator(
    node_id="cv2.warpAffine",
    default_render_options={"data": {"src": "out"}},
)
def warpAffine(img: ImageFormat, M: np.ndarray, w: int, h: int) -> OpenCVImageFormat:
    return OpenCVImageFormat(cv2.warpAffine(assert_opencvdata(img), M, (w, h)))


@fn.NodeDecorator(
    node_id="cv2.perpectiveTransform",
    default_render_options={"data": {"src": "out"}},
)
def perpectiveTransform(
    img: ImageFormat, M: np.ndarray, w: int, h: int
) -> OpenCVImageFormat:
    return OpenCVImageFormat(cv2.warpPerspective(assert_opencvdata(img), M, (w, h)))


@fn.NodeDecorator(
    node_id="cv2.getAffineTransform_points",
)
def getAffineTransform_points(
    i1x1: int,
    i1y1: int,
    i1x2: int,
    i1y2: int,
    i1x3: int,
    i1y3: int,
    i2x1: int,
    i2y1: int,
    i2x2: int,
    i2y2: int,
    i2x3: int,
    i2y3: int,
) -> np.ndarray:
    return cv2.getAffineTransform(
        np.array([[i1x1, i1y1], [i1x2, i1y2], [i1x3, i1y3]], dtype=np.float32),
        np.array([[i2x1, i2y1], [i2x2, i2y2], [i2x3, i2y3]], dtype=np.float32),
    )


@fn.NodeDecorator(
    node_id="cv2.getPerspectiveTransform_points",
)
def getPerspectiveTransform_points(
    i1x1: int,
    i1y1: int,
    i1x2: int,
    i1y2: int,
    i1x3: int,
    i1y3: int,
    i1x4: int,
    i1y4: int,
    i2x1: int,
    i2y1: int,
    i2x2: int,
    i2y2: int,
    i2x3: int,
    i2y3: int,
    i2x4: int,
    i2y4: int,
) -> np.ndarray:
    return cv2.getPerspectiveTransform(
        np.array(
            [[i1x1, i1y1], [i1x2, i1y2], [i1x3, i1y3], [i1x4, i1y4]], dtype=np.float32
        ),
        np.array(
            [[i2x1, i2y1], [i2x2, i2y2], [i2x3, i2y3], [i2x4, i2y4]], dtype=np.float32
        ),
    )


@fn.NodeDecorator(
    node_id="cv2.getAffineTransform",
)
def getAffineTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))


@fn.NodeDecorator(
    node_id="cv2.getPerspectiveTransform",
)
def getPerspectiveTransform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))


NODE_SHELF = fn.Shelf(
    name="Image Operations",
    description="Operations on images.",
    subshelves=[],
    nodes=[
        resize,
        flip,
        rotate,
        freeRotation,
        warpAffine,
        perpectiveTransform,
        getAffineTransform_points,
        getPerspectiveTransform_points,
        getAffineTransform,
        getPerspectiveTransform,
    ],
)
