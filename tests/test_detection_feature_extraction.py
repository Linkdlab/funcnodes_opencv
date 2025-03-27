import numpy as np
import cv2
import pytest
import pytest_funcnodes
from funcnodes_opencv.image_processing.detection_feature_extraction import (
    HoughCircles,
    HoughLines,
    HoughLinesP,
)
from pathlib import Path


def get_image_data(image_format):
    return image_format.data if hasattr(image_format, "data") else image_format


@pytest.fixture
def image():
    return cv2.GaussianBlur(
        cv2.imread(Path(__file__).parent / "astronaut.jpg", cv2.IMREAD_COLOR),
        (21, 21),
        2,
    )


@pytest_funcnodes.nodetest(HoughLines)
async def test_HoughLines(image):
    d, ang, c, lines = await HoughLines.inti_call(img=image)

    dlines = np.array(
        cv2.HoughLines(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 10, np.pi / 180, 200)
    )

    np.testing.assert_array_equal(lines, dlines)


@pytest_funcnodes.nodetest(HoughLinesP)
async def test_HoughLinesP(image):
    x1y1x2y2, x1y1, x2y2 = await HoughLinesP.inti_call(img=image)

    dlines = np.array(
        cv2.HoughLinesP(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 10, np.pi / 180, 200)
    )

    np.testing.assert_array_equal(x1y1x2y2, dlines)


@pytest_funcnodes.nodetest(HoughCircles)
async def test_HoughCircles(image):
    xy, r, votes, res = await HoughCircles.inti_call(img=image)

    dcircles = np.array(
        cv2.HoughCircles(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1.5, 20
        )
    )[0]

    np.testing.assert_array_equal(res, dcircles)

    # display_image = image.copy()
    # for _xy, _r in zip(xy, r):
    #     cv2.circle(display_image, _xy.astype(int), int(_r), (0, 255, 0), 1)
    #     cv2.circle(display_image, _xy.astype(int), 2, (0, 0, 255), 1)

    # cv2.imshow("Image", display_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
