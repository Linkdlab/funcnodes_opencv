import numpy as np
import cv2
import pytest
import pytest_funcnodes

from funcnodes_opencv.image_processing.thresholding import (
    threshold,
    auto_threshold,
    adaptive_threshold,
    in_range_singel_channel,
    in_range,
    AutoThresholdTypes,
    AdaptiveThresholdMethods,
)
from funcnodes_opencv.utils import assert_opencvdata


@pytest_funcnodes.nodetest(threshold)
@pytest.mark.parametrize(
    "thresh",
    np.linspace(0.1, 0.9, 4),
)
async def test_threshold(image1, thresh):
    res = cv2.threshold(
        image1.raw_transformed, int(thresh * 255), 255, cv2.THRESH_BINARY
    )[1]
    res = assert_opencvdata(res)
    fnout = (await threshold.inti_call(img=image1, thresh=thresh)).data

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=1)


@pytest_funcnodes.nodetest(auto_threshold)
@pytest.mark.parametrize(
    "type",
    list(AutoThresholdTypes),
)
async def test_auto_threshold(image1, type):
    img = image1.raw_transformed
    if image1.testchannels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cv2.threshold(img, 0, 255, type.value)[1]
    res = assert_opencvdata(res)
    fnout = (
        await auto_threshold.inti_call(
            img=image1,
            type=type,
        )
    )[0].data
    fnout = assert_opencvdata(fnout)
    # showdat([image1], res, fno ut)
    diff = np.abs(fnout - res)
    assert diff.mean() < 5e-3


@pytest_funcnodes.nodetest(adaptive_threshold)
@pytest.mark.parametrize(
    "type",
    list(AdaptiveThresholdMethods),
)
async def test_adaptive_threshold(image1, type):
    img = image1.raw_transformed
    if image1.testchannels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    res = cv2.adaptiveThreshold(
        img,
        255,
        type.value,
        cv2.THRESH_BINARY,
        2 * 5 + 1,
        255 * 0.2,
    )
    res = assert_opencvdata(res)
    fnout = (
        await adaptive_threshold.inti_call(img=image1, method=type, block_size=5, c=0.2)
    ).data
    fnout = assert_opencvdata(fnout)
    # showdat([image1], res, fnout)
    diff = np.abs(fnout - res)
    assert diff.mean() < 5e-2


@pytest_funcnodes.nodetest(in_range_singel_channel)
async def test_in_range_singel_channel(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    arr = cv2.inRange(img, np.array([50]), np.array([200]))
    res = assert_opencvdata(arr)
    fnout = (
        await in_range_singel_channel.inti_call(
            img=image1,
            lower=50 / 255,
            upper=200 / 255,
        )
    ).data

    # showdat([image1], res, fnout)
    diff = np.abs(fnout - res)
    assert diff.mean() < 1e-2


@pytest_funcnodes.nodetest(in_range)
async def test_in_range(image1):
    img = image1.raw_transformed.copy()
    if image1.testchannels == 1:
        arr = cv2.inRange(img, np.array([0]), np.array([200]))
    else:
        arr = cv2.inRange(
            img,
            np.array([0, 10, 100]),
            np.array([200, 100, 255]),
        )

    res = assert_opencvdata(arr)
    fnout = (
        await in_range.inti_call(
            img=image1,
            lower_c1=0 / 255,
            upper_c1=200 / 255,
            lower_c2=10 / 255,
            upper_c2=100 / 255,
            lower_c3=100 / 255,
            upper_c3=255 / 255,
        )
    ).data

    # showdat([image1], res, fnout)
    diff = np.abs(fnout - res)
    assert diff.mean() < 6e-3
