import numpy as np
import cv2
import pytest
import pytest_funcnodes

from funcnodes_opencv.image_operations.normalization_equalization import (
    normalize,
    equalizeHist,
    CLAHE,
)
from funcnodes_opencv.utils import assert_opencvdata


@pytest_funcnodes.nodetest(normalize)
async def test_normalize(image1):
    fnout = (await normalize.inti_call(img=image1))[0].data

    res = cv2.normalize(image1.raw_transformed, None, 0, 255, cv2.NORM_MINMAX)
    res = assert_opencvdata(res)

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=2e-3)


@pytest_funcnodes.nodetest(equalizeHist)
async def test_equalizeHist(image1):
    fnout = (await equalizeHist.inti_call(img=image1)).data

    img = image1.raw_transformed
    if image1.testchannels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.equalizeHist(img)
    res = assert_opencvdata(res)

    # showdat([image1], res, fnout)
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=2e-2)


@pytest_funcnodes.nodetest(CLAHE)
@pytest.mark.parametrize(
    "clipLimit",
    np.linspace(2, 90, 5, endpoint=True),
)
async def test_CLAHE(image1, clipLimit):
    img = image1.raw_transformed
    img = img.astype(np.uint16) * 255
    if image1.testchannels == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(8, 8)).apply(img)
    res = assert_opencvdata(res)

    fnout = (await CLAHE.inti_call(img=image1, clip_limit=clipLimit)).data

    # showdat([image1], res, fnout, title=f"clipLimit={clipLimit}")
    np.testing.assert_allclose(fnout, res, rtol=1e-6, atol=5e-2)
