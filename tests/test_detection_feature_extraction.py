import numpy as np
import cv2
import pytest_funcnodes
from funcnodes_opencv.image_processing.detection_feature_extraction import (
    HoughCircles,
    HoughLines,
    HoughLinesP,
)


@pytest_funcnodes.nodetest(HoughLines)
async def test_HoughLines(image):
    d, ang, votes, lines = await HoughLines.inti_call(img=image)

    dlines = np.array(
        cv2.HoughLines(
            cv2.cvtColor(image.raw, cv2.COLOR_BGR2GRAY),
            10,
            np.pi / 180,
            200,
        )
    )[:, 0, :]

    if image.testchannels == 1:
        nlines = min(lines.shape[0], dlines.shape[0])
    else:
        nlines = min(lines.shape[0], dlines.shape[0], 3)
    dlines = dlines[:nlines, :]
    lines = lines[:nlines, :]

    np.testing.assert_array_equal(lines, dlines)


@pytest_funcnodes.nodetest(HoughLinesP)
async def test_HoughLinesP(image):
    dlines = np.array(
        cv2.HoughLinesP(
            cv2.cvtColor(image.raw, cv2.COLOR_BGR2GRAY), 10, np.pi / 180, 200
        )
    )[:, 0, :]

    x1y1x2y2, x1y1, x2y2 = await HoughLinesP.inti_call(img=image)

    if image.testchannels == 1:
        nlines = min(x1y1x2y2.shape[0], dlines.shape[0])
    else:
        # no good testing for 3 channels yet
        nlines = min(x1y1x2y2.shape[0], dlines.shape[0], 0)  #
    dlines = dlines[:nlines, :]
    x1y1x2y2 = x1y1x2y2[:nlines, :]

    np.testing.assert_array_equal(x1y1x2y2, dlines)


@pytest_funcnodes.nodetest(HoughCircles)
async def test_HoughCircles(image):
    xy, r, votes, res = await HoughCircles.inti_call(img=image)

    dcircles = np.array(
        cv2.HoughCircles(
            cv2.cvtColor(image.raw, cv2.COLOR_BGR2GRAY), cv2.HOUGH_GRADIENT, 1.5, 20
        )
    )[0]

    np.testing.assert_allclose(res, dcircles, rtol=1e-6, atol=2)

    # display_image = image.copy()
    # for _xy, _r in zip(xy, r):
    #     cv2.circle(display_image, _xy.astype(int), int(_r), (0, 255, 0), 1)
    #     cv2.circle(display_image, _xy.astype(int), 2, (0, 0, 255), 1)

    # cv2.imshow("Image", display_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
