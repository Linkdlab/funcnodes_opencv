from typing import Tuple
import cv2
import numpy as np
import funcnodes as fn
from ..imageformat import ImageFormat
from ..utils import assert_opencvdata


# hughlines


@fn.NodeDecorator(
    node_id="cv2.HoughLines",
    default_render_options={"data": {"src": "out"}},
    description="Finds lines in a binary image using the standard Hough transform.",
    default_io_options={
        "theta": {"value_options": {"min": 0, "max": 180}},
        "min_theta": {"value_options": {"min": 0, "max": 180}},
        "max_theta": {"value_options": {"min": 0, "max": 180}},
    },
    outputs=[
        {
            "name": "d",
        },
        {
            "name": "ang",
        },
        {
            "name": "votes",
        },
        {"name": "lines"},
    ],
)
def HoughLines(
    img: ImageFormat,
    rho: float = 10,  # distance resolution in pixels
    theta: float = 1,  # angle resolution in degrees
    threshold: int = 200,  # accumulator threshold parameter
    srn: int = 0,  # For the multi-scale Hough transform, it is a divisor for the distance resolution rho
    stn: int = 0,  # For the multi-scale Hough transform, it is a divisor for the distance resolution theta
    min_theta: float = 0,  # For standard and multi-scale Hough transform, minimum angle to check for lines
    max_theta: float = 180,  # For standard and multi-scale Hough transform, maximum angle to check for lines
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    theta_rad = np.deg2rad(theta)
    max_theta_rad = np.deg2rad(max_theta)
    min_theta_rad = np.deg2rad(min_theta)
    res = np.array(
        cv2.HoughLines(
            assert_opencvdata(img, channel=1),
            rho,
            theta_rad,
            threshold,
            srn=srn,
            stn=stn,
            min_theta=min_theta_rad,
            max_theta=max_theta_rad,
        )
    )
    print("AAA", res.shape)
    d = res[:, 0, 0]
    ang = res[:, 0, 1]
    if res.shape[2] == 3:
        votes = res[:, 0, 2]
    else:
        votes = np.ones(res.shape[0])
    return d, ang, votes, res


# hughlinesp
@fn.NodeDecorator(
    node_id="cv2.HoughLinesP",
    default_render_options={"data": {"src": "out"}},
    description="Finds line segments in a binary image using the probabilistic Hough transform.",
    default_io_options={
        "theta": {"value_options": {"min": 0, "max": 180}},
        "min_line_length": {"value_options": {"min": 0}},
        "max_line_gap": {"value_options": {"min": 0}},
    },
    outputs=[
        {
            "name": "x1y1x2y2",
        },
        {
            "name": "x1y1",
        },
        {
            "name": "x2y2",
        },
    ],
)
def HoughLinesP(
    img: ImageFormat,
    rho: float = 10,  # distance resolution in pixels
    theta: float = 1,  # angle resolution in degrees
    threshold: int = 200,  # accumulator threshold parameter
    min_line_length: int = 0,  # minimum line length
    max_line_gap: int = 0,  # maximum allowed gap between points on the same line
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    theta_rad = np.deg2rad(theta)
    res = np.array(
        cv2.HoughLinesP(
            assert_opencvdata(img, channel=1),
            rho,
            theta_rad,
            threshold,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
    )
    x1y1 = res[:, :2]
    x2y2 = res[:, 2:]
    return res, x1y1, x2y2


# hughcircles
@fn.NodeDecorator(
    node_id="cv2.HoughCircles",
    default_render_options={"data": {"src": "out"}},
    description="Finds circles in a grayscale image using the Hough transform.",
    default_io_options={
        "dp": {"value_options": {"min": 1}},
        "min_dist": {"value_options": {"min": 0}},
        "param1": {"value_options": {"min": 0}},
        "param2": {"value_options": {"min": 0}},
        "min_radius": {"value_options": {"min": 0}},
        "max_radius": {"value_options": {"min": 0}},
    },
    outputs=[
        {
            "name": "xy",
        },
        {
            "name": "radii",
        },
        {
            "name": "votes",
        },
        {"name": "circles"},
    ],
)
def HoughCircles(
    img: ImageFormat,
    dp: float = 1.5,  # Inverse ratio of the accumulator resolution to the image resolution
    min_dist: float = 20,  # Minimum distance between the centers of the detected circles
    param1: int = 100,  # First method-specific parameter
    param2: int = 100,  # Second method-specific parameter
    min_radius: int = 0,  # Minimum circle radius
    max_radius: int = 0,  # Maximum circle radius
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    res = np.array(
        cv2.HoughCircles(
            assert_opencvdata(img, channel=1),
            cv2.HOUGH_GRADIENT,
            dp,
            minDist=min_dist,
            param1=param1,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
    )[0]

    xy = res[:, :2]
    r = res[:, 2]
    if res.shape[1] == 4:
        votes = res[:, 3]
    else:
        votes = np.ones(res.shape[0])
    return xy, r, votes, res


NODE_SHELF = fn.Shelf(
    name="Detection & Feature Extraction",
    description="OpenCV detection and feature extraction nodes.",
    nodes=[HoughLines, HoughLinesP, HoughCircles],
    subshelves=[],
)
